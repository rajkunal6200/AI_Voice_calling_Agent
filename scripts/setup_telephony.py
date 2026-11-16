"""Wire a phone number to the agent.

    uv run python -m scripts.setup_telephony --show
    uv run python -m scripts.setup_telephony --inbound +14155550123
    uv run python -m scripts.setup_telephony --outbound +14155550123 \
        --sip-address my-trunk.pstn.twilio.com --sip-user U --sip-pass P

Buying the number is the one step that cannot be scripted — do that in the
LiveKit Cloud dashboard (Telephony → Phone Numbers), or bring one from a
provider. Everything after it is here.

Inbound needs two objects: a *trunk* that owns the number, and a *dispatch
rule* that decides what happens when it rings. The rule is where calls are
routed to the agent, and its ``agent_name`` must match the name the worker
registers under — a mismatch is the classic failure where the call connects to
silence, because the room is created and nobody joins it.
"""

from __future__ import annotations

import argparse
import asyncio

from livekit import api

from agent.main import AGENT_NAME
from app.errors import ConfigurationError
from app.observability.logging import configure_logging, get_logger

log = get_logger(__name__)


def _client() -> api.LiveKitAPI:
    from app.config import settings

    if not settings.livekit_api_key:
        raise ConfigurationError("LIVEKIT_API_KEY is required")
    return api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )


async def show(lk: api.LiveKitAPI) -> None:
    inbound = await lk.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())
    rules = await lk.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())
    outbound = await lk.sip.list_outbound_trunk(api.ListSIPOutboundTrunkRequest())

    print(f"\n  worker registers as: {AGENT_NAME}\n")
    print(f"  inbound trunks  ({len(inbound.items)})")
    for t in inbound.items:
        print(f"    {t.sip_trunk_id}  {list(t.numbers)}  {t.name!r}")
    print(f"  dispatch rules  ({len(rules.items)})")
    for r in rules.items:
        agents = [a.agent_name for a in r.room_config.agents] if r.room_config else []
        match = "ok" if AGENT_NAME in agents else "MISMATCH — calls will hear silence"
        print(f"    {r.sip_dispatch_rule_id}  agents={agents}  {match}")
    print(f"  outbound trunks ({len(outbound.items)})")
    for t in outbound.items:
        print(f"    {t.sip_trunk_id}  {list(t.numbers)}  {t.name!r}")
    print()

    if inbound.items and not rules.items:
        print("  A trunk with no dispatch rule will ring and go nowhere.")
        print("  Run with --inbound <number> to create the rule.\n")


async def setup_inbound(lk: api.LiveKitAPI, number: str) -> None:
    """Create the trunk and the rule that routes its calls to the agent."""
    existing = await lk.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())
    trunk = next((t for t in existing.items if number in t.numbers), None)

    if trunk is None:
        trunk = await lk.sip.create_inbound_trunk(
            api.CreateSIPInboundTrunkRequest(
                trunk=api.SIPInboundTrunkInfo(
                    name=f"inbound {number}",
                    numbers=[number],
                    # Krisp removes background noise before speech recognition
                    # sees it, which matters more on a phone line than anywhere
                    # else — callers ring from cars and open-plan offices.
                    krisp_enabled=True,
                )
            )
        )
        log.info("inbound_trunk_created", trunk_id=trunk.sip_trunk_id, number=number)
    else:
        log.info("inbound_trunk_exists", trunk_id=trunk.sip_trunk_id)

    rules = await lk.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())
    if any(trunk.sip_trunk_id in r.trunk_ids for r in rules.items):
        log.info("dispatch_rule_exists", note="not creating a second one")
        return

    rule = await lk.sip.create_dispatch_rule(
        api.CreateSIPDispatchRuleRequest(
            name=f"route {number} to {AGENT_NAME}",
            trunk_ids=[trunk.sip_trunk_id],
            # One room per call. Without this every caller lands in the same
            # room and hears each other, which is a memorable way to find out.
            rule=api.SIPDispatchRule(
                dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                    room_prefix="call-"
                )
            ),
            # The line that makes the agent actually join. AGENT_NAME is
            # imported rather than typed, so it cannot drift from the worker.
            room_config=api.RoomConfiguration(
                agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)]
            ),
        )
    )
    log.info("dispatch_rule_created", rule_id=rule.sip_dispatch_rule_id)
    print(f"\n  {number} now rings {AGENT_NAME}.")
    print("  Start the worker with: uv run python -m agent.main start\n")


async def setup_outbound(
    lk: api.LiveKitAPI, number: str, address: str, username: str, password: str
) -> None:
    """Create the trunk outbound calls dial through."""
    trunk = await lk.sip.create_outbound_trunk(
        api.CreateSIPOutboundTrunkRequest(
            trunk=api.SIPOutboundTrunkInfo(
                name=f"outbound {number}",
                address=address,
                numbers=[number],
                auth_username=username,
                auth_password=password,
            )
        )
    )
    log.info("outbound_trunk_created", trunk_id=trunk.sip_trunk_id)
    print(f"\n  Add this to .env:\n\n    SIP_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="print current config")
    parser.add_argument("--inbound", metavar="E164", help="number to receive calls on")
    parser.add_argument("--outbound", metavar="E164", help="number to call from")
    parser.add_argument("--sip-address", help="provider SIP host, for --outbound")
    parser.add_argument("--sip-user", help="provider SIP username, for --outbound")
    parser.add_argument("--sip-pass", help="provider SIP password, for --outbound")
    args = parser.parse_args()

    configure_logging()
    lk = _client()
    try:
        if args.inbound:
            await setup_inbound(lk, args.inbound)
        if args.outbound:
            if not (args.sip_address and args.sip_user and args.sip_pass):
                raise ConfigurationError(
                    "--outbound needs --sip-address, --sip-user and --sip-pass "
                    "from your telephony provider"
                )
            await setup_outbound(
                lk, args.outbound, args.sip_address, args.sip_user, args.sip_pass
            )
        if args.show or not (args.inbound or args.outbound):
            await show(lk)
    finally:
        await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())
