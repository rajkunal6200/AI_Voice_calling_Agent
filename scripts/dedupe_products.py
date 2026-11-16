"""Merge products that exist twice under different SKUs.

``app/db.py`` used to seed a small catalogue of its own alongside the one in
``scripts/demo_data.py``, so a database created before that was fixed has three
products under two SKUs each. Stock split across two rows for one real product
means neither row is true: check_stock can report on one while place_order
decrements the other.

Keeps the SKU that ``demo_data`` knows about, moves any order lines onto it,
and deletes the stray. Safe to re-run — it finds nothing the second time.

    uv run python -m scripts.dedupe_products --dry-run
    uv run python -m scripts.dedupe_products
"""

from __future__ import annotations

import argparse
import asyncio

from app.db import engine, execute_returning, fetch_all
from app.observability.logging import configure_logging, get_logger
from scripts.demo_data import PRODUCTS

log = get_logger(__name__)


async def find_duplicates() -> list[dict[str, object]]:
    """Names with more than one SKU, and which SKU the demo dataset owns."""
    known = {p.sku for p in PRODUCTS}
    rows = await fetch_all(
        """
        SELECT name, array_agg(sku ORDER BY sku) AS skus
        FROM products GROUP BY name HAVING COUNT(*) > 1
        """
    )
    duplicates = []
    for row in rows:
        skus = list(row["skus"])
        keep = next((s for s in skus if s in known), skus[0])
        duplicates.append(
            {"name": row["name"], "keep": keep, "drop": [s for s in skus if s != keep]}
        )
    return duplicates


async def merge(duplicate: dict[str, object], *, dry_run: bool) -> None:
    keep = str(duplicate["keep"])
    for stray in duplicate["drop"]:  # type: ignore[union-attr]
        lines = await fetch_all(
            "SELECT order_ref FROM order_items WHERE sku = :sku", sku=stray
        )
        print(f"  {duplicate['name']:<26} {stray} -> {keep}", end="")
        print(f"  ({len(lines)} order line(s))" if lines else "")
        if dry_run:
            continue
        # Order lines first: deleting the product row would leave them naming
        # a SKU that no longer exists. There is no foreign key to stop it.
        if lines:
            await execute_returning(
                "UPDATE order_items SET sku = :keep WHERE sku = :stray "
                "RETURNING order_ref",
                keep=keep,
                stray=stray,
            )
        await execute_returning(
            "DELETE FROM products WHERE sku = :sku RETURNING sku", sku=stray
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show, change nothing")
    args = parser.parse_args()
    configure_logging()

    duplicates = await find_duplicates()
    if not duplicates:
        print("\n  No duplicate products. Nothing to do.\n")
        await engine.dispose()
        return

    print(f"\n  {len(duplicates)} product(s) under more than one SKU:\n")
    for duplicate in duplicates:
        await merge(duplicate, dry_run=args.dry_run)

    if args.dry_run:
        print("\n  Dry run — nothing changed. Re-run without --dry-run.\n")
    else:
        remaining = await fetch_all("SELECT COUNT(*) AS n FROM products")
        print(f"\n  Done. {remaining[0]['n']} products, all distinct.\n")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
