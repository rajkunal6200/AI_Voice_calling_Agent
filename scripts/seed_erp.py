"""Load the demo catalogue and orders into Postgres.

Run: ``uv run python -m scripts.seed_erp``          # add what is missing
     ``uv run python -m scripts.seed_erp --reset``  # wipe and reload

Pair it with ``scripts.seed_hubspot`` — the two share the same customer list,
and the phone numbers must match for caller identification and order lookup to
agree.

This used to wipe and reload unconditionally, which was harmless while orders
were only ever seeded. They are not any more: ``place_order`` writes real ones
during a call, and a caller who has been read an order reference should still
have that order tomorrow. Re-seeding also resets ``contact_phone`` on every
order, which silently undoes a number change made by ``save_caller_details``.
So the destructive path is opt-in, and the default only adds what is missing.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from app.db import engine
from app.observability.logging import configure_logging, get_logger
from scripts.demo_data import CUSTOMERS, PRODUCTS, all_orders

log = get_logger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete every product and order first, including real ones",
    )
    args = parser.parse_args()
    configure_logging()

    async with engine.begin() as conn:
        if args.reset:
            # order_items cascades from orders; products are independent.
            await conn.execute(text("DELETE FROM orders"))
            await conn.execute(text("DELETE FROM products"))

        for product in PRODUCTS:
            await conn.execute(
                text(
                    """
                    INSERT INTO products (sku, name, price_cents, stock_qty)
                    VALUES (:sku, :name, :price, :qty)
                    ON CONFLICT (sku) DO NOTHING
                    """
                ),
                {
                    "sku": product.sku,
                    "name": product.name,
                    "price": product.price_cents,
                    "qty": product.stock_qty,
                },
            )

        for customer, order in all_orders():
            await conn.execute(
                text(
                    """
                    INSERT INTO orders (
                        order_ref, contact_email, contact_phone, status,
                        total_cents, tracking_ref, eta
                    )
                    VALUES (
                        :ref, :email, :phone, :status, :total, :tracking,
                        -- Cast is required: a bound parameter arrives untyped
                        -- and `date + unknown` is ambiguous to Postgres.
                        CURRENT_DATE + CAST(:eta_days AS integer)
                    )
                    ON CONFLICT (order_ref) DO NOTHING
                    """
                ),
                {
                    "ref": order.order_ref,
                    "email": customer.email,
                    "phone": customer.phone,
                    "status": order.status,
                    "total": order.total_cents,
                    "tracking": order.tracking_ref,
                    "eta_days": order.eta_days,
                },
            )
            for sku, qty in order.items:
                await conn.execute(
                    text(
                        "INSERT INTO order_items (order_ref, sku, qty) "
                        "SELECT :ref, :sku, :qty WHERE NOT EXISTS ("
                        "  SELECT 1 FROM order_items "
                        "  WHERE order_ref = :ref AND sku = :sku)"
                    ),
                    {"ref": order.order_ref, "sku": sku, "qty": qty},
                )

    in_stock = sum(1 for p in PRODUCTS if p.stock_qty > 0)
    log.info(
        "erp_seeded",
        reset=args.reset,
        products=len(PRODUCTS),
        in_stock=in_stock,
        out_of_stock=len(PRODUCTS) - in_stock,
        customers=len(CUSTOMERS),
        orders=len(all_orders()),
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
