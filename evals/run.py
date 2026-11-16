"""Score the agent's behaviour against ``evals/cases.py``.

    uv run python -m evals.run
    uv run python -m evals.run --model openai/gpt-4.1-mini   # compare models
    uv run python -m evals.run --repeat 3                    # sample the variance

Runs each case through a real ``AgentSession`` with a real model, but no room,
no audio and no phone — text in, tool calls and text out. That makes it fast
enough to run after every prompt change, and cheap enough to run three times to
see how stable the behaviour actually is.

The output is a score, not a pass/fail, because the thing being measured is not
deterministic. Use it to compare: this prompt against that one, this model
against that one. A single run in isolation tells you much less than the
difference between two runs.

These write. ``sales.commit`` places a real order and consumes real stock, so
the runner points itself at a throwaway database — see ``evals/scratch_db.py``.
They also write to whatever CRM ``HUBSPOT_ACCESS_TOKEN`` names, which has no
equivalent escape hatch: expect deals and notes to appear in that portal.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass

from dotenv import load_dotenv

# Must come before anything under `app.`: importing it sets DATABASE_URL, and
# app.config builds its Settings — app.db its engine — at import time. The
# cases write real orders and consume real stock, which does not belong in the
# development database.
from evals import scratch_db  # noqa: E402  (import order is load-bearing)

load_dotenv()

from livekit.agents import AgentSession, inference  # noqa: E402

from agent import entry_agent  # noqa: E402
from agent.state import CallState  # noqa: E402
from agent.tooling import current_call_id  # noqa: E402
from app.config import settings  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.services.hubspot import Contact  # noqa: E402
from evals.cases import CASES, Case  # noqa: E402
from scripts.demo_data import CUSTOMERS  # noqa: E402

#: A seeded customer, so the cases run as a caller the CRM recognises.
DEMO_PHONE = CUSTOMERS[0].phone


@dataclass
class Outcome:
    case: Case
    passed: bool
    elapsed_ms: int
    tools_called: list[str]
    final_agent: str
    reply: str
    failures: list[str]


async def run_case(case: Case, *, model: str) -> Outcome:
    # A fresh session per case: carrying conversation state between unrelated
    # utterances would make each result depend on the ones before it.
    current_call_id.set(f"eval-{case.id}")
    # A known caller, as on a real call from a seeded number. Without a contact
    # the agent correctly asks for details it does not have, which is a
    # different scenario from the one most of these cases are about.
    state = CallState(
        call_id=f"eval-{case.id}",
        caller_phone=DEMO_PHONE,
        contact=Contact(
            id="101",
            first_name="Ada",
            last_name="Okonkwo",
            email="ada.okonkwo@example.com",
            phone=DEMO_PHONE,
        ),
    )
    session = AgentSession[CallState](
        userdata=state,
        llm=inference.LLM(model=model),
    )
    await session.start(agent=entry_agent(state))

    started = time.monotonic()
    try:
        result = await session.run(user_input=case.utterance)
    finally:
        elapsed_ms = int((time.monotonic() - started) * 1000)

    tools: list[str] = []
    handoffs: list[str] = []
    reply_parts: list[str] = []
    for event in result.events:
        kind = type(event).__name__
        if kind == "FunctionCallEvent":
            tools.append(getattr(event.item, "name", "?"))
        elif kind == "AgentHandoffEvent":
            handoffs.append(type(getattr(event, "new_agent", None)).__name__)
        elif kind == "ChatMessageEvent":
            item = getattr(event, "item", None)
            if getattr(item, "role", None) == "assistant":
                reply_parts.append(getattr(item, "text_content", "") or "")

    reply = " ".join(p for p in reply_parts if p).strip()
    final_agent = handoffs[-1] if handoffs else "ReceptionAgent"

    failures: list[str] = []
    if case.expect_tool and case.expect_tool not in tools:
        failures.append(f"expected tool {case.expect_tool}, got {tools or 'none'}")
    if case.expect_any_tool and not set(case.expect_any_tool) & set(tools):
        failures.append(
            f"expected one of {case.expect_any_tool}, got {tools or 'none'}"
        )
    for forbidden in case.forbid_tools:
        if forbidden in tools:
            failures.append(f"must not have called {forbidden}")
    lowered = reply.lower()
    for phrase in case.expect_says:
        if phrase.lower() not in lowered:
            failures.append(f"reply missing {phrase!r}")
    for phrase in case.forbid_says:
        if phrase.lower() in lowered:
            failures.append(f"reply should not contain {phrase!r}")
    if case.grounded_by is not None:
        phrases, required_tool = case.grounded_by
        if any(p.lower() in lowered for p in phrases) and required_tool not in tools:
            failures.append(
                f"stated {phrases} without calling {required_tool} — invented"
            )

    await session.aclose()
    return Outcome(
        case=case,
        passed=not failures,
        elapsed_ms=elapsed_ms,
        tools_called=tools,
        final_agent=final_agent,
        reply=reply,
        failures=failures,
    )


def render(outcomes: list[Outcome], *, model: str, repeat: int) -> bool:
    by_case: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_case.setdefault(o.case.id, []).append(o)

    from app.config import settings as _s
    print(
        f"\n  model: {model}    mode: {_s.agent_mode}    "
        f"cases: {len(by_case)}    runs each: {repeat}\n"
    )
    print(f"  {'case':<28} {'score':>7}  {'p50 ms':>7}  detail")
    print(f"  {'-' * 28} {'-' * 7}  {'-' * 7}  {'-' * 40}")

    total_passed = 0
    for case_id, runs in by_case.items():
        passed = sum(r.passed for r in runs)
        total_passed += passed
        times = sorted(r.elapsed_ms for r in runs)
        p50 = times[len(times) // 2]
        mark = "ok " if passed == len(runs) else ("~  " if passed else "FAIL")
        detail = ""
        if passed < len(runs):
            first_failure = next(r for r in runs if not r.failures == [])
            detail = first_failure.failures[0][:56]
        print(f"  {case_id:<28} {mark} {passed}/{len(runs)}  {p50:>7}  {detail}")

    total = len(outcomes)
    rate = total_passed / total if total else 0.0
    print(f"\n  {total_passed}/{total} passed  ({rate:.0%})\n")

    # Show what actually happened on failures — the reply is usually the
    # fastest way to see whether the model misread the request or the prompt.
    for case_id, runs in by_case.items():
        bad = [r for r in runs if not r.passed]
        if not bad:
            continue
        first = bad[0]
        print(f"  {case_id}: {first.case.because}")
        print(f"    said : {first.reply[:150] or '(nothing)'}")
        print(f"    tools: {first.tools_called or 'none'} -> {first.final_agent}\n")

    return rate == 1.0


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="runs per case; >1 shows how stable the behaviour is",
    )
    parser.add_argument("--only", help="run cases whose id contains this substring")
    args = parser.parse_args()

    configure_logging()
    await scratch_db.prepare()
    cases = [c for c in CASES if not args.only or args.only in c.id]

    outcomes: list[Outcome] = []
    for case in cases:
        for _ in range(args.repeat):
            outcomes.append(await run_case(case, model=args.model))

    everything_passed = render(outcomes, model=args.model, repeat=args.repeat)
    raise SystemExit(0 if everything_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
