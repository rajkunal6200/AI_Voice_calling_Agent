"""Voice agent worker.

Run with ``uv run python -m agent.main dev`` (or ``start`` in production).

One session per phone call. The entrypoint's job is to establish who is calling
before the agent says anything, so the greeting can use their name — that
lookup is the difference between "thanks for calling" and "hi Ada".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    inference,
)

from agent import entry_agent
from agent.outbound import OutboundQualificationAgent
from agent.state import CallState
from agent.tooling import current_call_id
from app.config import settings
from app.observability.logging import bind_call_context, configure_logging, get_logger
from app.observability.sentry import capture, init_error_reporting
from app.services import calls, knowledge
from app.services.hubspot import Contact, HubSpotClient

load_dotenv()
configure_logging()
init_error_reporting()

log = get_logger(__name__)

def _prewarm(proc: Any) -> None:
    """Load the embedding model before any call is assigned to this process.

    It is a few seconds on first load. Paying that lazily means the first
    caller to ask about the returns policy hears several seconds of silence,
    so it happens here instead — synchronously, before the process takes work.
    """
    try:
        knowledge.load_model()
    except Exception as exc:  # noqa: BLE001 — retrieval degrades, calls still work
        log.warning("embedding_prewarm_failed", error=str(exc))


server = AgentServer(setup_fnc=_prewarm)

#: Both the inbound SIP dispatch rule and the outbound campaign runner target
#: this name. AgentServer supports one registered session, so one name.
AGENT_NAME = "voice-agent"

# LiveKit populates these on the participant for calls arriving over SIP.
SIP_PHONE_ATTRIBUTE = "sip.phoneNumber"
SIP_CALL_ID_ATTRIBUTE = "sip.callID"


def _caller_phone(participant: rtc.RemoteParticipant) -> str | None:
    """The caller's number, or None for a browser/test participant."""
    attributes: dict[str, str] = dict(participant.attributes or {})
    phone = attributes.get(SIP_PHONE_ATTRIBUTE)
    if phone:
        return phone
    # Console and browser participants carry no SIP attributes. Fall back to
    # the configured number so caller identification can be exercised without
    # a phone line; on a real call the attribute above always wins.
    if settings.console_caller_phone:
        log.info(
            "using_console_caller_phone",
            phone=settings.console_caller_phone,
            note="no SIP attributes; simulating a known caller",
        )
        return settings.console_caller_phone

    log.info(
        "no_sip_attributes",
        identity=participant.identity,
        note="not a SIP call and no CONSOLE_CALLER_PHONE; caller is unknown",
    )
    return None


async def _identify_caller(
    hubspot: HubSpotClient | None, phone: str | None
) -> Contact | None:
    """Best-effort CRM lookup.

    Deliberately never raises. A CRM outage should degrade the greeting from
    "hi Ada" to "thanks for calling" — it should not drop the call, which is
    what an exception here would do before the agent ever speaks.
    """
    if hubspot is None or not phone:
        return None
    try:
        return await hubspot.find_contact_by_phone(phone)
    except Exception as exc:  # noqa: BLE001 — see docstring
        log.error("caller_identification_failed", error=str(exc))
        capture(exc, stage="caller_identification")
        return None


def _build_session(state: CallState) -> AgentSession[CallState]:
    # userdata carries CallState across handoffs — specialists are swapped out
    # mid-call, so anything held on an agent instance would be lost.
    return AgentSession[CallState](
        userdata=state,
        stt=inference.STT(model=settings.stt_model, language=settings.stt_language),
        llm=inference.LLM(model=settings.llm_model),
        tts=inference.TTS(model=settings.tts_model, voice=settings.tts_voice),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            # Every model request costs ~1.5s of network round trip from here,
            # so the only way to feel faster is to overlap that with something.
            #
            # Preemptive generation starts the model while the caller is still
            # finishing their sentence, and preemptive TTS starts speaking
            # before the full response is generated. Together they hide most of
            # one round trip behind speech that was happening anyway. The cost
            # is wasted generations when someone pauses mid-sentence and then
            # continues — tokens, not correctness.
            preemptive_generation={"enabled": True, "preemptive_tts": True},
            # How long to wait after the caller stops talking before deciding
            # the turn is over. This is pure dead air, and the default is
            # tuned for conversational English rather than for a link with an
            # already-large round trip. Lower it and the agent feels sharper;
            # too low and it interrupts people who pause to think.
            endpointing={"min_delay": settings.endpointing_min_delay},
        ),
    )


def _wire_transcripts(
    session: AgentSession[CallState], call_id: str, state: CallState
) -> None:
    """Persist the conversation and any runtime errors to the call timeline.

    LiveKit emits these events synchronously but the writes are async, so each
    handler spawns a task. The tasks are held in a set until they finish:
    asyncio only keeps a weak reference to a running task, so a bare
    ``create_task`` can be garbage-collected mid-write and silently lose the
    row. Failures inside ``record_event`` are swallowed there — a timeline
    write must never disturb a live call.
    """
    pending: set[asyncio.Task[None]] = set()

    # The most recent per-component timings, attached to the agent's next
    # utterance. Metrics and transcripts arrive on separate events, and only
    # the utterance row is a place a reader would think to look for "how long
    # did this turn take" — so they are joined here rather than in the reader.
    turn: dict[str, int] = {}

    def _spawn(coro: Any) -> None:
        try:
            task = asyncio.create_task(coro)
        except RuntimeError:  # no running loop, e.g. during teardown
            coro.close()
            return
        pending.add(task)
        task.add_done_callback(pending.discard)

    @session.on("conversation_item_added")
    def _on_item(event: Any) -> None:
        item = event.item
        role = getattr(item, "role", None)
        kind = {"user": "user_speech", "assistant": "agent_speech"}.get(str(role))
        if kind is None:
            return
        if kind == "user_speech":
            # The caller spoke, so any handoffs so far were in service of a
            # request they made rather than a loop between agents.
            state.handoffs_since_user_turn = 0
        text = getattr(item, "text_content", None) or ""
        payload: dict[str, Any] = {
            "text": text,
            "interrupted": getattr(item, "interrupted", False),
        }
        latency_ms: int | None = None
        if kind == "agent_speech" and turn:
            # What the caller actually experienced: the silence from them
            # finishing to the agent starting. The breakdown rides along in the
            # payload, because the fixes for a slow turn detector, a slow model
            # and slow speech synthesis are three different fixes.
            payload["latency"] = dict(turn)
            latency_ms = (
                turn.get("end_of_utterance_ms", 0)
                + turn.get("llm_ttft_ms", 0)
                + turn.get("tts_ttfb_ms", 0)
            ) or None
            turn.clear()
        _spawn(
            calls.record_event(
                call_id=call_id,
                kind=kind,  # type: ignore[arg-type]
                payload=payload,
                latency_ms=latency_ms,
            )
        )

    # Per-component turn latency. Without this, "the agent takes 7 seconds" is
    # unactionable — the fix for a slow turn detector (config), a slow LLM
    # (model choice) and slow speech synthesis (voice choice) are all
    # different, and they are indistinguishable from the outside.
    @session.on("metrics_collected")
    def _on_metrics(event: Any) -> None:
        m = event.metrics
        kind = type(m).__name__

        if kind == "EOUMetrics":
            # How long we wait after the caller stops talking before deciding
            # the turn is over. Pure dead air, and often the biggest single
            # slice of perceived latency.
            turn["end_of_utterance_ms"] = round(m.end_of_utterance_delay * 1000)
            log.info(
                "latency_turn_detection",
                end_of_utterance_ms=turn["end_of_utterance_ms"],
                transcription_ms=round(m.transcription_delay * 1000),
            )
        elif kind == "LLMMetrics":
            turn["llm_ttft_ms"] = round(m.ttft * 1000)
            log.info(
                "latency_llm",
                ttft_ms=turn["llm_ttft_ms"],
                total_ms=round(m.duration * 1000),
                prompt_tokens=m.prompt_tokens,
                completion_tokens=m.completion_tokens,
                tokens_per_second=round(m.tokens_per_second, 1),
            )
        elif kind == "TTSMetrics":
            turn["tts_ttfb_ms"] = round(m.ttfb * 1000)
            log.info(
                "latency_tts",
                ttfb_ms=turn["tts_ttfb_ms"],
                total_ms=round(m.duration * 1000),
                characters=m.characters_count,
            )
        elif kind == "STTMetrics":
            log.info("latency_stt", duration_ms=round(m.duration * 1000))

    @session.on("error")
    def _on_error(event: Any) -> None:
        source = str(getattr(event, "source", ""))
        log.error("session_error", source=source, error=str(event.error))
        _spawn(
            calls.record_event(
                call_id=call_id,
                kind="error",
                name="session",
                payload={"error": str(event.error), "source": source},
            )
        )


@server.rtc_session(agent_name=AGENT_NAME)
async def voice_session(ctx: JobContext) -> None:
    """One entrypoint for both directions.

    AgentServer registers exactly one rtc_session, so direction is decided
    here. An inbound SIP job arrives from a dispatch rule with no metadata; an
    outbound job is created explicitly by the campaign runner and carries the
    number to dial. The presence of that number is the discriminator.
    """
    try:
        target = json.loads(ctx.job.metadata or "{}")
    except json.JSONDecodeError:
        log.error("bad_job_metadata", metadata=ctx.job.metadata)
        target = {}

    if target.get("phone"):
        await _outbound_session(ctx, target)
    else:
        await _inbound_session(ctx)


async def _inbound_session(ctx: JobContext) -> None:
    call_id = ctx.room.name
    current_call_id.set(call_id)
    bind_call_context(call_id=call_id, direction="inbound")

    await ctx.connect()
    participant = await ctx.wait_for_participant()
    phone = _caller_phone(participant)

    await calls.start_call(call_id=call_id, direction="inbound", from_number=phone)

    # Optional: without a token the agent still answers, checks stock, looks
    # up orders and books appointments — it just cannot identify the caller or
    # write back to the CRM.
    hubspot = HubSpotClient() if settings.hubspot_access_token else None
    if hubspot is None:
        log.warning("hubspot_not_configured", note="CRM lookups and writes disabled")
    contact = await _identify_caller(hubspot, phone)
    if contact:
        bind_call_context(contact_id=contact.id)
        await calls.attach_contact(
            call_id=call_id,
            hubspot_contact_id=contact.id,
            caller_name=contact.display_name,
        )

    log.info("session_starting", identified=contact is not None, has_phone=bool(phone))

    state = CallState(
        call_id=call_id, caller_phone=phone, hubspot=hubspot, contact=contact
    )
    session = _build_session(state)
    _wire_transcripts(session, call_id, state)

    async def _on_shutdown() -> None:
        # The agent normally closes the call itself via log_call_summary. This
        # covers hangups mid-conversation, where that never runs — without it,
        # abandoned calls would sit at 'in_progress' forever in the dashboard.
        with contextlib.suppress(Exception):
            current = await calls.get_call(call_id)
            if current.status == "in_progress" and not state.summary_logged:
                await calls.end_call(
                    call_id=call_id,
                    status="completed",
                    summary="Call ended before a summary was recorded.",
                )
        if hubspot is not None:
            await hubspot.aclose()
        log.info("session_shutdown")

    ctx.add_shutdown_callback(_on_shutdown)

    agent: Agent = entry_agent(state)
    await session.start(room=ctx.room, agent=agent)

    greeting = (
        f"Greet {contact.display_name} by name and ask how you can help."
        if contact
        else "Greet the caller warmly and ask how you can help."
    )
    await session.generate_reply(instructions=greeting)


async def _outbound_session(ctx: JobContext, target: dict[str, Any]) -> None:
    """Place a call and qualify the lead.

    Inverted from inbound: the job exists before anyone is on the line and we
    dial out. Who to call arrives as job metadata, written by
    ``scripts.run_campaign`` when it created the dispatch.
    """
    call_id = ctx.room.name
    current_call_id.set(call_id)
    bind_call_context(call_id=call_id, direction="outbound")

    phone = target["phone"]
    if not settings.sip_outbound_trunk_id:
        log.error("outbound_no_trunk", note="set SIP_OUTBOUND_TRUNK_ID to place calls")
        return

    await ctx.connect()
    await calls.start_call(call_id=call_id, direction="outbound", to_number=phone)

    hubspot = HubSpotClient() if settings.hubspot_access_token else None
    contact = await _identify_caller(hubspot, phone)
    state = CallState(
        call_id=call_id,
        caller_phone=phone,
        hubspot=hubspot,
        contact=contact,
        deal_id=target.get("deal_id"),
    )

    session = _build_session(state)
    _wire_transcripts(session, call_id, state)

    async def _on_shutdown() -> None:
        with contextlib.suppress(Exception):
            current = await calls.get_call(call_id)
            if current.status == "in_progress" and not state.summary_logged:
                # No answer, voicemail, or they hung up before we recorded an
                # outcome. Marked failed rather than completed so the campaign
                # can tell "not reached" from "reached and declined".
                await calls.end_call(
                    call_id=call_id,
                    status="failed",
                    summary="Call ended before an outcome was recorded.",
                )
        if hubspot is not None:
            await hubspot.aclose()
        log.info("outbound_session_shutdown")

    ctx.add_shutdown_callback(_on_shutdown)

    agent: Agent = OutboundQualificationAgent(state=state)
    await session.start(room=ctx.room, agent=agent)

    log.info("dialing", to=phone, deal_id=state.deal_id)
    # Blocks until the callee picks up. If they never do, the job ends and the
    # shutdown handler records it as unreached.
    await ctx.add_sip_participant(
        call_to=phone,
        trunk_id=settings.sip_outbound_trunk_id,
        participant_identity=f"caller-{call_id}",
    )

    who = contact.display_name if contact else "them"
    await session.generate_reply(
        instructions=(
            f"Introduce yourself to {who} and say why you are calling, then ask "
            "if it is a good time to talk."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
