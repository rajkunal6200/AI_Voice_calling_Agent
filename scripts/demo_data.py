"""The demo dataset, defined once and used by both seeders.

The ERP seeder writes products and orders to Postgres; the HubSpot seeder
writes the same people to the CRM. They share this module because the phone
numbers have to match exactly: caller identification looks a number up in
HubSpot, and order lookup matches the same number against Postgres. Seed them
from separate lists and you get a caller the agent greets by name but whose
order it cannot find — which looks like a bug in the agent rather than in the
fixtures.

Numbers are all +1555 01xx, the range reserved for fiction, so nothing here can
dial a real person.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price_cents: int
    stock_qty: int


@dataclass(frozen=True)
class Order:
    order_ref: str
    status: str  # processing | shipped | delivered | cancelled
    total_cents: int
    tracking_ref: str | None = None
    eta_days: int | None = None
    items: list[tuple[str, int]] = field(default_factory=list)  # (sku, qty)


@dataclass(frozen=True)
class Customer:
    first_name: str
    last_name: str
    email: str
    phone: str
    company: str | None = None
    orders: list[Order] = field(default_factory=list)


# --- catalogue -------------------------------------------------------------
# Deliberately mixed: things in stock, things low, and things at zero, so the
# agent has to report all three rather than always saying yes.

PRODUCTS: list[Product] = [
    # Desks
    Product("DSK-1001", "Ridgeline Standing Desk", 74900, 12),
    Product("DSK-1002", "Corner Workstation Desk", 52900, 4),
    Product("DSK-1003", "Compact Home Office Desk", 29900, 0),
    # Seating
    Product("CHR-2001", "Meridian Ergonomic Chair", 61900, 18),
    Product("CHR-2002", "Drafting Stool", 22900, 6),
    Product("CHR-2003", "Visitor Chair", 14900, 31),
    # Lighting
    Product("LGT-3001", "Aeron Desk Lamp", 12900, 42),
    Product("LGT-3002", "Clamp-On Task Light", 7900, 0),
    # Storage
    Product("STO-4001", "Three Drawer Pedestal", 24900, 9),
    Product("STO-4002", "Lockable Filing Cabinet", 38900, 2),
    # Accessories
    Product("ACC-5001", "Dual Monitor Arm", 18900, 25),
    Product("ACC-5002", "Standing Desk Mat", 6500, 0),
    Product("ACC-5003", "Mechanical Keyboard", 18900, 7),
    Product("ACC-5004", "Under-Desk Cable Tray", 4900, 60),
]


# --- customers and their orders --------------------------------------------

CUSTOMERS: list[Customer] = [
    Customer(
        first_name="Ada",
        last_name="Okonkwo",
        email="ada.okonkwo@example.com",
        phone="+12025550100",
        company="Brightline Studio",
        orders=[
            Order(
                "ORD-5001", "shipped", 87800, tracking_ref="TRK99213", eta_days=2,
                items=[("LGT-3001", 1), ("DSK-1002", 1)],
            ),
            Order(
                "ORD-5006", "delivered", 6500, tracking_ref="TRK88110",
                items=[("ACC-5002", 1)],
            ),
        ],
    ),
    Customer(
        first_name="Grace",
        last_name="Whitfield",
        email="grace.whitfield@example.com",
        phone="+12025550101",
        orders=[
            Order(
                "ORD-5002", "processing", 74900, eta_days=5,
                items=[("DSK-1001", 1)],
            ),
        ],
    ),
    Customer(
        first_name="Tomas",
        last_name="Reyes",
        email="tomas.reyes@example.com",
        phone="+12025550102",
        company="Reyes Architects",
        orders=[
            Order(
                "ORD-5003", "shipped", 123800, tracking_ref="TRK77420", eta_days=1,
                items=[("CHR-2001", 2)],
            ),
        ],
    ),
    Customer(
        first_name="Priya",
        last_name="Raman",
        email="priya.raman@example.com",
        phone="+12025550103",
        orders=[
            Order(
                "ORD-5004", "cancelled", 29900,
                items=[("DSK-1003", 1)],
            ),
        ],
    ),
    # No orders: exercises the "known caller, nothing to look up" path.
    Customer(
        first_name="Marcus",
        last_name="Bell",
        email="marcus.bell@example.com",
        phone="+12025550104",
        company="Bell & Co",
    ),
]


def all_orders() -> list[tuple[Customer, Order]]:
    return [(c, o) for c in CUSTOMERS for o in c.orders]
