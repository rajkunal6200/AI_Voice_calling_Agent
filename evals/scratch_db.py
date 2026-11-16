"""Point the evals at a throwaway database before anything imports ``app``.

Evals used to be nearly read-only, so running them against the development
database was harmless. That stopped being true when ``place_order`` arrived:
the ``sales.commit`` case places a real order and consumes real stock every
time it runs. Three repeats left six orders in the dev database and took twelve
lamps out of inventory, which then looked like demo data.

Importing this module has the side effect of setting ``DATABASE_URL``, so it
must be imported *before* anything under ``app.`` — ``app.config`` builds its
Settings and ``app.db`` its engine at import time. That ordering is the same
load-bearing trick ``tests/conftest.py`` uses, and for the same reason.

Set ``EVAL_DATABASE_URL`` to override, or point it at the dev database if you
deliberately want the side effects to land there.
"""

from __future__ import annotations

import os

DEFAULT_URL = "postgresql+asyncpg://postgres:postgres@localhost:5434/voice_crm_evals"

EVAL_DATABASE_URL = os.environ.get("EVAL_DATABASE_URL", DEFAULT_URL)
os.environ["DATABASE_URL"] = EVAL_DATABASE_URL

DATABASE_NAME = EVAL_DATABASE_URL.rsplit("/", 1)[-1]


async def prepare() -> None:
    """Create the database if needed, apply the schema, load the demo data.

    The cases assume a real catalogue: they ask for a desk lamp by name and
    expect a price. An empty database would score the model down for the
    harness having nothing to sell.
    """
    import asyncpg

    admin = await asyncpg.connect(
        user="postgres",
        password="postgres",
        host="localhost",
        port=5434,
        database="postgres",
    )
    try:
        exists = await admin.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", DATABASE_NAME
        )
        if not exists:
            await admin.execute(f'CREATE DATABASE "{DATABASE_NAME}"')
    finally:
        await admin.close()

    # Imported here, not at module scope: the environment above has to be set
    # before app.db is first imported anywhere in the process.
    from sqlalchemy import text

    from app.db import apply_schema, engine
    from scripts.demo_data import PRODUCTS

    await apply_schema()
    async with engine.begin() as conn:
        # Reset the catalogue each run so a previous run's purchases do not
        # slowly empty the shelves and change what the next run measures.
        await conn.execute(text("DELETE FROM orders"))
        await conn.execute(text("DELETE FROM products"))
        for product in PRODUCTS:
            await conn.execute(
                text(
                    "INSERT INTO products (sku, name, price_cents, stock_qty) "
                    "VALUES (:sku, :name, :price, :qty)"
                ),
                {
                    "sku": product.sku,
                    "name": product.name,
                    "price": product.price_cents,
                    "qty": product.stock_qty,
                },
            )
