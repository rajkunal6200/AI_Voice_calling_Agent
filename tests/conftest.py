"""Test fixtures.

Postgres is real, not mocked. Most of the invariants worth protecting here are
enforced *by the database* — the UNIQUE constraint that prevents double
booking, the CHECK constraints on status columns, the ON CONFLICT clauses — so
a mocked database would mock away the thing under test.

HubSpot is mocked, at the transport layer via respx. That keeps the retry and
error-mapping logic under test (which is ours) without depending on a live
account or a network round trip (which is not).

Requires the Postgres container: ``docker compose up -d``.
"""

from __future__ import annotations

import os

# Point at a dedicated test database *before* importing anything under app.,
# because app.config builds its Settings — and app.db its engine — at import.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5434/voice_crm_test",
)
os.environ.setdefault("SENTRY_DSN", "")
os.environ.setdefault("HUBSPOT_ACCESS_TOKEN", "test-token")

import asyncpg  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import apply_schema, engine  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402

TABLES = (
    "call_events",
    "calls",
    "order_items",
    "orders",
    "products",
    "appointments",
    "documents",
)


async def _ensure_database() -> None:
    """Create voice_crm_test if it isn't there yet."""
    admin = await asyncpg.connect(
        user="postgres", password="postgres", host="localhost", port=5434, database="postgres"
    )
    try:
        exists = await admin.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = 'voice_crm_test'"
        )
        if not exists:
            await admin.execute("CREATE DATABASE voice_crm_test")
    finally:
        await admin.close()


@pytest.fixture(scope="session", autouse=True)
async def database() -> None:
    configure_logging()
    await _ensure_database()
    # No seed rows: tests that need data insert exactly what they assert on,
    # so a change to the demo seed can never quietly break the suite.
    await apply_schema()
    yield
    await engine.dispose()


@pytest.fixture(autouse=True)
async def clean_tables() -> None:
    """Truncate between tests so ordering can never matter."""
    async with engine.begin() as conn:
        await conn.execute(
            text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
        )
    yield


@pytest.fixture
async def seed_products() -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO products (sku, name, price_cents, stock_qty) VALUES
                    ('SKU-1', 'Aeron Desk Lamp', 12900, 42),
                    ('SKU-2', 'Standing Desk Mat', 6500, 0)
                """
            )
        )


@pytest.fixture
async def seed_orders() -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO orders
                    (order_ref, contact_phone, status, total_cents, tracking_ref)
                VALUES
                    ('ORD-1', '+15551234567', 'shipped', 12900, 'TRK1'),
                    ('ORD-2', '+15551234567', 'processing', 6500, NULL)
                """
            )
        )
