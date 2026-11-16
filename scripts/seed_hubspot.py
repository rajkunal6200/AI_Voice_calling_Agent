"""Create the demo customers in HubSpot.

Run: ``uv run python -m scripts.seed_hubspot``

Writes the same people as ``scripts.seed_erp`` so the two systems agree: a
caller from +15550100 is identified as Ada in the CRM *and* has orders in the
ERP. Seed only one and the agent greets someone by name but cannot find their
order, which reads as an agent bug rather than a fixture mismatch.

Idempotent — contacts are matched by phone first, so re-running does not
duplicate anything. That matters because this writes to a real portal.

``--dry-run`` prints what it would do without touching HubSpot.
"""

from __future__ import annotations

import argparse
import asyncio

from app.errors import AgentError
from app.observability.logging import configure_logging, get_logger
from app.services.hubspot import HubSpotClient
from scripts.demo_data import CUSTOMERS

log = get_logger(__name__)

# Deals are only created for customers with an open (processing) order, so the
# outbound qualification agent has something realistic to work with.
DEAL_STATUSES = {"processing"}


async def seed(*, dry_run: bool) -> None:
    if dry_run:
        for customer in CUSTOMERS:
            open_orders = [o for o in customer.orders if o.status in DEAL_STATUSES]
            print(
                f"  {customer.first_name} {customer.last_name:<12} "
                f"{customer.phone:<14} {customer.email:<32} "
                f"{len(customer.orders)} order(s), {len(open_orders)} deal(s)"
            )
        print(f"\n{len(CUSTOMERS)} contacts would be created or matched.")
        return

    client = HubSpotClient()
    try:
        stage = await client.first_deal_stage()
        log.info("using_deal_stage", stage=stage)

        created = matched = deals = 0
        for customer in CUSTOMERS:
            contact, was_created = await client.upsert_contact(
                phone=customer.phone,
                first_name=customer.first_name,
                last_name=customer.last_name,
                email=customer.email,
                company=customer.company,
            )
            created += was_created
            matched += not was_created

            for order in customer.orders:
                if order.status not in DEAL_STATUSES:
                    continue
                name = f"{customer.first_name} {customer.last_name} — {order.order_ref}"
                # Contacts dedupe by phone; deals have no natural key, so
                # check by name or a re-run stacks duplicates in a real portal.
                if await client.find_deal_by_name(name) is not None:
                    continue
                await client.create_deal(
                    name=name,
                    amount_cents=order.total_cents,
                    stage=stage,
                    contact_id=contact.id,
                )
                deals += 1

        log.info(
            "hubspot_seeded",
            contacts_created=created,
            contacts_matched=matched,
            deals_created=deals,
        )
    finally:
        await client.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be written without calling HubSpot",
    )
    args = parser.parse_args()

    configure_logging()
    try:
        await seed(dry_run=args.dry_run)
    except AgentError as exc:
        # A bad token is the overwhelmingly likely cause, and the raw HubSpot
        # message ("expired 20677 days ago") does not point at it.
        log.error("hubspot_seed_failed", reason=exc.message, **exc.context)
        print(
            "\nHubSpot rejected the request. Check HUBSPOT_ACCESS_TOKEN is a "
            "service key or private app token, from HubSpot under Development "
            "> Keys > Service Keys, or Development > Legacy Apps.\n"
            "An OAuth access token (looks like 'CiRl...') will not work: those "
            "last about 30 minutes, and HubSpot reports the failure as "
            "'expired 20677 day(s) ago', which does not point at the cause."
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
