"""ERP: inventory and order status.

HubSpot models contacts and deals but has no notion of stock levels or
fulfilment state, so the ERP half runs here against Postgres. The point of the
seam is that the agent's tools call these functions and never touch SQL — when
a real ERP (Odoo, NetSuite) replaces this, the tool layer is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import text

from app.db import execute_returning, fetch_all, fetch_one, transaction
from app.errors import NotFoundError, UserFacingError, ValidationError
from app.observability.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class StockItem:
    sku: str
    name: str
    price_cents: int
    stock_qty: int

    @property
    def in_stock(self) -> bool:
        return self.stock_qty > 0

    @property
    def price_display(self) -> str:
        return f"${self.price_cents / 100:,.2f}"


@dataclass(frozen=True)
class Order:
    order_ref: str
    status: str
    total_cents: int
    tracking_ref: str | None
    eta: date | None


async def find_product(query: str) -> StockItem:
    """Look a product up by SKU or by name.

    Callers say "the desk lamp", not "SKU-1001", so name matching is a
    case-insensitive substring. Raises ``NotFoundError`` when nothing matches,
    which the tool layer turns into the agent saying so out loud.
    """
    row = await fetch_one(
        """
        SELECT sku, name, price_cents, stock_qty
        FROM products
        WHERE upper(sku) = upper(:query) OR name ILIKE :pattern
        ORDER BY (upper(sku) = upper(:query)) DESC, name
        LIMIT 1
        """,
        query=query,
        pattern=f"%{query}%",
    )
    if row is None:
        log.info("product_not_found", query=query)
        raise NotFoundError(
            f"no product matching {query!r}",
            caller_message="I couldn't find that product in our catalogue.",
            query=query,
        )
    item = StockItem(**row)
    log.info("product_found", sku=item.sku, stock_qty=item.stock_qty)
    return item


async def list_catalogue() -> list[StockItem]:
    """Everything we sell, in stock first.

    Callers open with "what do you sell?" far more often than with a product
    name, and there was no tool for that — check_stock needs something to look
    up. Without this the question had no answer anywhere, which is how two
    agents ended up passing it back and forth.
    """
    rows = await fetch_all(
        """
        SELECT sku, name, price_cents, stock_qty
        FROM products
        ORDER BY (stock_qty > 0) DESC, name
        """
    )
    log.info("catalogue_listed", count=len(rows))
    return [StockItem(**row) for row in rows]


async def relink_orders(*, old_phone: str, new_phone: str) -> int:
    """Move a customer's orders to their new number. Returns how many moved.

    The phone number is the join key between the CRM and this ERP: the caller
    is identified in HubSpot by number, and their orders are found here by the
    same number. Changing it in one place only leaves the caller greeted by
    name and told they have no orders, which looks like the lookup is broken.
    """
    rows = await execute_returning(
        """
        UPDATE orders SET contact_phone = :new_phone
        WHERE contact_phone = :old_phone
        RETURNING order_ref
        """,
        old_phone=old_phone,
        new_phone=new_phone,
    )
    log.info("orders_relinked", moved=len(rows), old=old_phone, new=new_phone)
    return len(rows)


async def place_order(
    *,
    sku: str,
    quantity: int,
    contact_phone: str | None,
    contact_email: str | None = None,
    hubspot_deal_id: str | None = None,
    eta_days: int = 5,
) -> Order:
    """Take stock out of inventory and create the order that consumed it.

    This is the step that was missing between the two systems. A deal is an
    opportunity in the CRM — someone *might* buy. An order is a commitment in
    the ERP, and until something wrote one, a deal could never become an order
    and every order in the database was seeded history with no origin.

    Stock and order are written in one transaction: reserving stock against an
    order that does not exist, or an order for goods never taken out of
    inventory, are both worse than the call failing. Concurrency is settled by
    the database the same way double-booking is — the decrement carries its own
    ``stock_qty >= quantity`` guard, so two callers racing for the last item
    means one UPDATE matches no row rather than both reading "in stock" first.
    """
    if quantity < 1:
        raise ValidationError(
            f"quantity {quantity} is not orderable",
            caller_message="I can only order one or more of something.",
            quantity=quantity,
        )

    async with transaction() as session:
        # Price from the row we are about to decrement, inside the same
        # transaction — a price read earlier in the call may be stale.
        product = (
            await session.execute(
                text(
                    "SELECT sku, name, price_cents, stock_qty FROM products "
                    "WHERE upper(sku) = upper(:sku)"
                ),
                {"sku": sku},
            )
        ).mappings().first()
        if product is None:
            raise NotFoundError(
                f"no product {sku!r}",
                caller_message="I couldn't find that product to order.",
                sku=sku,
            )

        taken = (
            await session.execute(
                text(
                    "UPDATE products SET stock_qty = stock_qty - :quantity "
                    "WHERE sku = :sku AND stock_qty >= :quantity "
                    "RETURNING stock_qty"
                ),
                {"sku": product["sku"], "quantity": quantity},
            )
        ).mappings().first()
        if taken is None:
            # The product exists — it was read a statement ago — so the only
            # way the guard fails is not enough stock.
            raise UserFacingError(
                f"only {product['stock_qty']} of {product['sku']} in stock",
                caller_message=(
                    f"I can only order {product['stock_qty']} of those right now."
                    if product["stock_qty"]
                    else "That one's out of stock at the moment."
                ),
                sku=product["sku"],
                requested=quantity,
                available=product["stock_qty"],
            )

        order_ref = "ORD-{}".format(
            (await session.execute(text("SELECT nextval('order_ref_seq')"))).scalar_one()
        )
        total_cents = product["price_cents"] * quantity

        row = (
            await session.execute(
                text(
                    """
                    INSERT INTO orders (
                        order_ref, contact_email, contact_phone, status,
                        total_cents, eta, hubspot_deal_id
                    )
                    VALUES (
                        :order_ref, :contact_email, :contact_phone, 'processing',
                        :total_cents, CURRENT_DATE + CAST(:eta_days AS integer),
                        :hubspot_deal_id
                    )
                    RETURNING order_ref, status, total_cents, tracking_ref, eta
                    """
                ),
                {
                    "order_ref": order_ref,
                    "contact_email": contact_email,
                    "contact_phone": contact_phone,
                    "total_cents": total_cents,
                    "eta_days": eta_days,
                    "hubspot_deal_id": hubspot_deal_id,
                },
            )
        ).mappings().one()

        await session.execute(
            text(
                "INSERT INTO order_items (order_ref, sku, qty) "
                "VALUES (:order_ref, :sku, :qty)"
            ),
            {"order_ref": order_ref, "sku": product["sku"], "qty": quantity},
        )

    order = Order(**row)
    log.info(
        "order_placed",
        order_ref=order.order_ref,
        sku=product["sku"],
        quantity=quantity,
        total_cents=order.total_cents,
        deal_id=hubspot_deal_id,
        stock_left=taken["stock_qty"],
    )
    return order


async def find_order_by_deal(hubspot_deal_id: str) -> Order | None:
    """The order a deal turned into, if it turned into one.

    The reverse of the link ``place_order`` writes, so a deal in the CRM can be
    traced to its fulfilment without going through the contact.
    """
    row = await fetch_one(
        """
        SELECT order_ref, status, total_cents, tracking_ref, eta
        FROM orders WHERE hubspot_deal_id = :deal_id
        """,
        deal_id=hubspot_deal_id,
    )
    return Order(**row) if row else None


async def get_order(order_ref: str) -> Order:
    row = await fetch_one(
        """
        SELECT order_ref, status, total_cents, tracking_ref, eta
        FROM orders
        WHERE upper(order_ref) = upper(:order_ref)
        """,
        order_ref=order_ref,
    )
    if row is None:
        log.info("order_not_found", order_ref=order_ref)
        raise NotFoundError(
            f"no order {order_ref!r}",
            caller_message="I couldn't find an order with that reference.",
            order_ref=order_ref,
        )
    order = Order(**row)
    log.info("order_found", order_ref=order.order_ref, status=order.status)
    return order


async def find_orders_for_phone(phone: str) -> list[Order]:
    """Recent orders for a caller, newest first.

    Lets the agent answer "where's my order?" without the caller reciting a
    reference number — the common case on an inbound support line.
    """
    rows = await fetch_all(
        """
        SELECT order_ref, status, total_cents, tracking_ref, eta
        FROM orders
        WHERE contact_phone = :phone
        ORDER BY placed_at DESC
        LIMIT 5
        """,
        phone=phone,
    )
    log.info("orders_for_phone", phone=phone, count=len(rows))
    return [Order(**row) for row in rows]
