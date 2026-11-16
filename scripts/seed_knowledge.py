"""Index the policy documents the agent can answer from.

Separate from ``python -m app.db`` because indexing loads the embedding model,
and schema creation should stay fast and dependency-light.

Run: ``uv run python -m scripts.seed_knowledge``

Each entry is deliberately one short paragraph. Passages are read aloud, so a
long chunk becomes a monologue no caller can follow — chunk at the size you
would want spoken.
"""

from __future__ import annotations

import asyncio

from app.db import engine
from app.observability.logging import configure_logging, get_logger
from app.services import knowledge

log = get_logger(__name__)

POLICIES: dict[str, str] = {
    "Returns policy": (
        "Customers can return any item within 30 days of delivery for a full "
        "refund, as long as it is unused and in its original packaging. "
        "Assembled furniture can be returned within 14 days but is subject to "
        "a 15 percent restocking fee. Refunds reach the original payment "
        "method within five working days of us receiving the item."
    ),
    "Warranty": (
        "All desks and chairs carry a five year structural warranty. Lamps, "
        "monitor arms and accessories carry a two year warranty. The warranty "
        "covers manufacturing defects and frame failure, but not accidental "
        "damage, normal wear, or damage caused by incorrect assembly."
    ),
    "Delivery": (
        "Standard delivery is free on orders over 500 dollars and takes three "
        "to five working days. Next day delivery costs 25 dollars if ordered "
        "before 2pm. We deliver to the ground floor as standard; upstairs "
        "delivery can be arranged for an extra 40 dollars."
    ),
    "Assembly service": (
        "Assembly costs 60 dollars per item and is booked as an appointment. "
        "Our fitters work weekdays between 9am and 5pm and need roughly one "
        "hour per desk. Booking assembly does not affect the warranty; "
        "self-assembly is fine too, and instructions ship with every item."
    ),
    "Opening hours": (
        "The phone line is open Monday to Friday, 9am to 5pm. The showroom is "
        "open Monday to Saturday, 10am to 6pm, and closed on Sundays and "
        "public holidays."
    ),
    "Business accounts": (
        "Businesses ordering ten or more items qualify for trade pricing and "
        "30 day payment terms. A trade account needs a company registration "
        "number and takes two working days to approve."
    ),
}


async def main() -> None:
    configure_logging()
    await knowledge.warm_up()
    for title, body in POLICIES.items():
        await knowledge.index_document(title=title, body=body)
    log.info("knowledge_seeded", documents=len(POLICIES))
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
