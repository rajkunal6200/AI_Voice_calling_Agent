"""Database engine and schema.

Raw SQL, no ORM models. The queries here are small and the shapes the agent
needs are query-specific enough that an ORM would add indirection without
removing much code.

``python -m app.db`` applies the schema; every statement is ``IF NOT EXISTS``,
so it only ever creates and is safe to re-run. The first time a column needs to
change type or a constraint needs dropping, that is the signal to bring in
Alembic rather than to bolt an ALTER onto this list.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.observability.logging import configure_logging, get_logger

log = get_logger(__name__)

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    # A call that dies mid-flight should not hold a connection open; recycle
    # well under Postgres' idle timeout.
    pool_recycle=1800,
)

session_factory = async_sessionmaker(engine, expire_on_commit=False)


SCHEMA_STATEMENTS: list[str] = [
    # --- Knowledge base -----------------------------------------------------
    # Policy prose the ERP tables cannot answer ("what's your returns policy").
    # Lives in the same database as everything else, so retrieval needs no
    # extra service. Requires the pgvector image — see docker-compose.yml.
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS documents (
        id        BIGSERIAL PRIMARY KEY,
        title     TEXT UNIQUE NOT NULL,
        body      TEXT NOT NULL,
        embedding vector(384),
        indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # HNSW over cosine distance, matching the `<=>` operator the search uses.
    # At this corpus size a sequential scan would be fine; the index is here so
    # the query plan does not change shape when the corpus grows.
    """
    CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
    """,
    # --- Call history: the observable record of what the agent did ----------
    # HubSpot is the system of record for the *business* outcome; this is the
    # record of the *conversation*, which HubSpot has no good place for and
    # which the dashboard reads.
    """
    CREATE TABLE IF NOT EXISTS calls (
        id              BIGSERIAL PRIMARY KEY,
        call_id         TEXT UNIQUE NOT NULL,
        direction       TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
        from_number     TEXT,
        to_number       TEXT,
        hubspot_contact_id TEXT,
        status          TEXT NOT NULL DEFAULT 'in_progress'
                        CHECK (status IN ('in_progress', 'completed', 'failed')),
        summary         TEXT,
        started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        ended_at        TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS calls_started_at_idx ON calls (started_at DESC)",
    # Append-only timeline. One row per utterance, tool call, tool result and
    # error, so the dashboard can replay a call and so a failed call can be
    # debugged after the caller has hung up.
    """
    CREATE TABLE IF NOT EXISTS call_events (
        id          BIGSERIAL PRIMARY KEY,
        call_id     TEXT NOT NULL,
        kind        TEXT NOT NULL CHECK (kind IN (
                        'user_speech', 'agent_speech',
                        'tool_call', 'tool_result', 'error'
                    )),
        name        TEXT,
        payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
        latency_ms  INTEGER,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS call_events_call_id_idx ON call_events (call_id, id)",
    "CREATE INDEX IF NOT EXISTS call_events_kind_idx ON call_events (kind, created_at DESC)",
    # --- ERP: inventory and orders -----------------------------------------
    # Stands in for the ERP half that HubSpot does not model. Deliberately
    # behind the same service interface as the CRM side so it can be swapped
    # for Odoo/NetSuite without the agent tools changing.
    """
    CREATE TABLE IF NOT EXISTS products (
        sku         TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        price_cents INTEGER NOT NULL CHECK (price_cents >= 0),
        stock_qty   INTEGER NOT NULL DEFAULT 0 CHECK (stock_qty >= 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_ref     TEXT PRIMARY KEY,
        contact_email TEXT,
        contact_phone TEXT,
        status        TEXT NOT NULL DEFAULT 'processing'
                      CHECK (status IN ('processing', 'shipped', 'delivered', 'cancelled')),
        total_cents   INTEGER NOT NULL DEFAULT 0,
        tracking_ref  TEXT,
        placed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        eta           DATE,
        -- The deal this order came from, when the agent placed it. Null for
        -- orders that arrived by any other route, which is most of them.
        hubspot_deal_id TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS orders_contact_phone_idx ON orders (contact_phone)",
    # Order references the agent hands out. Starts above the seeded ORD-50xx
    # block so a demo order never collides with demo history, and a sequence
    # rather than max()+1 so two concurrent calls cannot pick the same ref.
    "CREATE SEQUENCE IF NOT EXISTS order_ref_seq START WITH 6000",
    """
    CREATE TABLE IF NOT EXISTS order_items (
        id        BIGSERIAL PRIMARY KEY,
        order_ref TEXT NOT NULL REFERENCES orders(order_ref) ON DELETE CASCADE,
        sku       TEXT NOT NULL,
        qty       INTEGER NOT NULL CHECK (qty > 0)
    )
    """,
    # --- Scheduling ---------------------------------------------------------
    # A booked slot. The UNIQUE constraint on starts_at is what makes
    # double-booking impossible under concurrency — two agents racing on the
    # same slot means one INSERT loses, rather than both reading "free" first.
    """
    CREATE TABLE IF NOT EXISTS appointments (
        id                 BIGSERIAL PRIMARY KEY,
        call_id            TEXT,
        hubspot_contact_id TEXT,
        starts_at          TIMESTAMPTZ NOT NULL UNIQUE,
        duration_minutes   INTEGER NOT NULL DEFAULT 30,
        notes              TEXT,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
]


# Changes to tables that already exist somewhere. The CREATE statements above
# only ever run against a fresh database, so a column added to one of them is
# invisible to every database that already has the table — which is the one
# case the "only ever creates" rule does not cover.
#
# `ADD COLUMN IF NOT EXISTS` is still idempotent, so this list keeps the
# property that matters: re-running the module is safe. It does not keep the
# other one — there is no ordering, no history, and no down path. That is
# affordable for a nullable column with no backfill and nothing else. Anything
# needing a type change, a NOT NULL, a rename or a data migration is where this
# stops working and Alembic starts.
# Anything depending on a migrated column belongs here too, not in the list
# above: on an existing database that column does not exist until this list
# runs, so an index over it in SCHEMA_STATEMENTS fails outright.
MIGRATION_STATEMENTS: list[str] = [
    "ALTER TABLE orders ADD COLUMN IF NOT EXISTS hubspot_deal_id TEXT",
    "CREATE INDEX IF NOT EXISTS orders_deal_idx ON orders (hubspot_deal_id)",
    # Who the caller was, as they were known at the time. A snapshot, not a
    # lookup: the dashboard showed a bare number and a contact id, which is
    # unreadable, and resolving names live would mean a CRM round trip per row.
    # Denormalised deliberately, for the same reason from_number is — it
    # records this call, and must not change when the contact does.
    "ALTER TABLE calls ADD COLUMN IF NOT EXISTS caller_name TEXT",
]


async def apply_schema() -> None:
    """Create the schema. Demo rows come from ``scripts.seed_erp``, not here.

    This module used to seed a few products and orders of its own, which was
    convenient until the demo dataset grew its own home in
    ``scripts/demo_data.py``. Two seeders then described the same catalogue
    differently: three products existed under two SKUs each, and ORD-5001 was
    written twice with different phone numbers, so whichever ran first won.
    Split stock across two rows for one real product and neither row is true.
    """
    async with engine.begin() as conn:
        for statement in SCHEMA_STATEMENTS:
            await conn.execute(text(statement))
        for statement in MIGRATION_STATEMENTS:
            await conn.execute(text(statement))
    log.info(
        "schema_applied",
        statements=len(SCHEMA_STATEMENTS),
        migrations=len(MIGRATION_STATEMENTS),
    )


async def fetch_all(sql: str, **params: Any) -> list[dict[str, Any]]:
    async with session_factory() as session:
        result = await session.execute(text(sql), params)
        return [dict(row) for row in result.mappings()]


async def fetch_one(sql: str, **params: Any) -> dict[str, Any] | None:
    async with session_factory() as session:
        result = await session.execute(text(sql), params)
        row = result.mappings().first()
        return dict(row) if row else None


async def execute_returning(sql: str, **params: Any) -> list[dict[str, Any]]:
    """Run a writing statement that returns rows, and commit it.

    ``fetch_all`` does not commit — it is for reads. An ``UPDATE ... RETURNING``
    run through it reports the rows it changed and then discards them when the
    session closes, which looks like success everywhere except the database.
    """
    async with session_factory() as session:
        result = await session.execute(text(sql), params)
        rows = [dict(row) for row in result.mappings()]
        await session.commit()
        return rows


@asynccontextmanager
async def transaction() -> AsyncIterator[AsyncSession]:
    """A session whose statements commit together or not at all.

    The other helpers here each open their own session, which is right for a
    single statement and wrong for several that depend on each other. Placing
    an order writes the order, its lines and a stock decrement; any one of
    those landing without the others is worse than the order failing outright
    — stock reserved against an order that does not exist, or an order for
    goods that were never taken out of inventory.
    """
    async with session_factory() as session:
        async with session.begin():
            yield session


async def execute(sql: str, **params: Any) -> None:
    async with session_factory() as session:
        await session.execute(text(sql), params)
        await session.commit()


async def _main() -> None:
    configure_logging()
    await apply_schema()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
