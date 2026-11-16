"""Place outbound qualification calls against open deals in HubSpot.

    uv run python -m scripts.run_campaign --dry-run    # who would be called
    uv run python -m scripts.run_campaign --limit 1    # call one lead
    uv run python -m scripts.run_campaign

Creates one agent dispatch per lead. The worker picks each up, dials, and runs
``OutboundQualificationAgent``; nothing here talks to anyone. That split means
a crashed campaign runner cannot drop a call already in progress.

Calls are placed one at a time with a gap between them. Dialling a list in
parallel is how a bug becomes a hundred unwanted calls before anyone notices,
and unsolicited calls are the kind of mistake that cannot be taken back.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from livekit import api

from app.config import settings
from app.errors import ConfigurationError
from app.observability.logging import configure_logging, get_logger
from app.services.hubspot import HubSpotClient

log = get_logger(__name__)

#: Seconds between calls. Deliberately unhurried — this is a rate limit on
#: mistakes as much as on the phone system.
DELAY_BETWEEN_CALLS = 20.0

#: Only deals in these stages are followed up.
TARGET_STAGES = {"appointmentscheduled", "qualifiedtobuy"}


async def open_deals(client: HubSpotClient, *, limit: int) -> list[dict[str, str]]:
    """Deals worth calling, with the contact to call for each.

    A deal with no associated contact has nobody to ring, so it is skipped
    rather than guessed at.
    """
    data = await client._request(  # noqa: SLF001 — campaign-only query
        "POST",
        "/crm/v3/objects/deals/search",
        json={
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "dealstage",
                            "operator": "IN",
                            "values": sorted(TARGET_STAGES),
                        }
                    ]
                }
            ],
            "properties": ["dealname", "dealstage", "amount"],
            "limit": limit,
        },
    )

    leads: list[dict[str, str]] = []
    for deal in data.get("results", []):
        deal_id = str(deal["id"])
        assoc = await client._request(  # noqa: SLF001
            "GET", f"/crm/v4/objects/deals/{deal_id}/associations/contacts"
        )
        results = assoc.get("results") or []
        if not results:
            log.warning("deal_has_no_contact", deal_id=deal_id)
            continue

        contact_id = str(results[0]["toObjectId"])
        contact = await client._request(  # noqa: SLF001
            "GET",
            f"/crm/v3/objects/contacts/{contact_id}?properties=phone,firstname,lastname",
        )
        phone = (contact.get("properties") or {}).get("phone")
        if not phone:
            log.warning("contact_has_no_phone", contact_id=contact_id)
            continue

        props = contact.get("properties") or {}
        leads.append(
            {
                "deal_id": deal_id,
                "deal_name": deal["properties"].get("dealname", ""),
                "contact_id": contact_id,
                "phone": phone,
                "name": f"{props.get('firstname', '')} {props.get('lastname', '')}".strip(),
            }
        )
    return leads


async def dispatch_call(lk: api.LiveKitAPI, lead: dict[str, str]) -> str:
    """Create the job. The worker does the dialling."""
    room = f"outbound-{uuid.uuid4().hex[:10]}"
    await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name="voice-agent",
            room=room,
            # The worker branches on this: a phone number here means outbound.
            metadata=json.dumps(
                {
                    "phone": lead["phone"],
                    "deal_id": lead["deal_id"],
                    "contact_id": lead["contact_id"],
                }
            ),
        )
    )
    log.info("call_dispatched", room=room, to=lead["phone"], deal_id=lead["deal_id"])
    return room


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list leads, call nobody")
    parser.add_argument("--limit", type=int, default=5, help="maximum calls to place")
    args = parser.parse_args()

    configure_logging()

    if not settings.hubspot_access_token:
        raise ConfigurationError("HUBSPOT_ACCESS_TOKEN is required to find leads")

    client = HubSpotClient()
    try:
        leads = await open_deals(client, limit=args.limit)
    finally:
        await client.aclose()

    if not leads:
        print("No open deals with a contact and phone number. Nothing to call.")
        return

    print(f"\n  {len(leads)} lead(s) to call:\n")
    for lead in leads:
        print(f"    {lead['name']:<20} {lead['phone']:<16} {lead['deal_name']}")

    if args.dry_run:
        print("\n  --dry-run: no calls placed.\n")
        return

    if not settings.sip_outbound_trunk_id:
        raise ConfigurationError(
            "SIP_OUTBOUND_TRUNK_ID is required to place calls. Create an "
            "outbound trunk with `lk sip outbound create`."
        )

    print()
    lk = api.LiveKitAPI(
        url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    try:
        for index, lead in enumerate(leads):
            await dispatch_call(lk, lead)
            print(f"  calling {lead['name']} on {lead['phone']}")
            # Serial, with a gap: a bug that dials in parallel becomes a
            # hundred unwanted calls before anyone can stop it.
            if index < len(leads) - 1:
                await asyncio.sleep(DELAY_BETWEEN_CALLS)
    finally:
        await lk.aclose()

    print(f"\n  {len(leads)} call(s) dispatched. Watch them at localhost:5173\n")


if __name__ == "__main__":
    asyncio.run(main())
