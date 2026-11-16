"""Measure where turn latency goes, so tuning is guided rather than guessed.

    uv run python -m scripts.measure_latency
    uv run python -m scripts.measure_latency --model openai/gpt-4.1-mini

Measures the text path only — no STT, no TTS, no turn detection — which
isolates the model round trips. Those dominate: on a link with ~1.5s to the
inference endpoint, a turn costs one round trip per model call, and the count
depends entirely on how many decisions the turn requires.

The first turn of a call is the expensive one, because routing through
reception adds decisions that later turns on the same specialist do not repeat.
Knowing that split is what tells you whether to attack the model, the network,
or the conversation structure.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from dotenv import load_dotenv
from livekit.agents import AgentSession, inference

from agent import entry_agent
from agent.state import CallState
from agent.tooling import current_call_id
from app.config import settings
from app.observability.logging import configure_logging

load_dotenv()

#: A first turn that must route, then two that stay on the same specialist.
CONVERSATION = [
    ("Do you have the desk lamp in stock?", "first turn"),
    ("What about the mechanical keyboard?", "follow-up"),
    ("And the dual monitor arm?", "follow-up"),
]

#: Rough per-round-trip cost, used only to show the arithmetic. Measure your
#: own with `scripts/measure_latency.py --raw`.
ROUND_TRIP_HINT_MS = 1500


async def measure(model: str) -> None:
    current_call_id.set("latency-probe")
    state = CallState(call_id="latency-probe", caller_phone="+15550100")
    session = AgentSession[CallState](
        userdata=state, llm=inference.LLM(model=model)
    )
    await session.start(agent=entry_agent(state))

    print(f"\n  model: {model}\n")
    print(f"  {'turn':<6} {'ms':>7}  {'tools':<34} note")
    print(f"  {'-' * 6} {'-' * 7}  {'-' * 34} {'-' * 26}")

    timings: list[int] = []
    for index, (utterance, note) in enumerate(CONVERSATION, start=1):
        started = time.monotonic()
        result = await session.run(user_input=utterance)
        elapsed = int((time.monotonic() - started) * 1000)
        timings.append(elapsed)
        tools = [
            getattr(e.item, "name", "?")
            for e in result.events
            if type(e).__name__ == "FunctionCallEvent"
        ]
        print(f"  {index:<6} {elapsed:>7}  {', '.join(tools):<34} {note}")

    await session.aclose()

    first, *rest = timings
    if rest:
        steady = sum(rest) // len(rest)
        print(f"\n  first turn : {first:>6} ms   (includes routing)")
        print(f"  thereafter : {steady:>6} ms   ({first / steady:.1f}x faster)")
        print(
            f"\n  Routing costs roughly {first - steady} ms, once per call. "
            f"At ~{ROUND_TRIP_HINT_MS}ms per model round trip, that is the\n"
            "  network, not the model — swapping models does not move it, "
            "and neither does prompt size.\n"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=settings.llm_model)
    args = parser.parse_args()
    configure_logging()
    await measure(args.model)


if __name__ == "__main__":
    asyncio.run(main())
