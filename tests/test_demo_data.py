"""The demo dataset must describe each thing exactly once.

``scripts/demo_data.py`` is the single source for the demo catalogue, orders
and customers, read by both seeders. It earned that role by failing at it:
``app/db.py`` seeded a small catalogue of its own for a while, so three
products ended up under two SKUs each and ORD-5001 was written twice with
different phone numbers.

Duplication there is not cosmetic. Callers name products, not SKUs, so
``erp.find_product`` matches on name — two rows sharing a name means it picks
one arbitrarily, and ``check_stock`` can report on one while ``place_order``
decrements the other. Stock split across two rows for one real product means
neither row is true.
"""

from __future__ import annotations

from collections import Counter

from scripts.demo_data import CUSTOMERS, PRODUCTS


def _duplicates(values: list[str]) -> list[str]:
    return sorted(v for v, n in Counter(values).items() if n > 1)


def test_every_sku_appears_once() -> None:
    assert _duplicates([p.sku for p in PRODUCTS]) == []


def test_no_two_products_share_a_name() -> None:
    """A caller says "the desk lamp", so the name has to identify one thing."""
    assert _duplicates([p.name for p in PRODUCTS]) == []


def test_every_order_reference_appears_once() -> None:
    refs = [o.order_ref for c in CUSTOMERS for o in c.orders]
    assert _duplicates(refs) == []


def test_no_two_customers_share_a_phone_number() -> None:
    """The number is the join key between the CRM and the ERP. Two customers
    on one number means identification cannot tell them apart."""
    assert _duplicates([c.phone for c in CUSTOMERS]) == []


def test_order_lines_reference_products_that_exist() -> None:
    """order_items.sku has no foreign key, so a typo here would seed an order
    line naming a product that is not in the catalogue."""
    catalogue = {p.sku for p in PRODUCTS}
    unknown = {
        sku
        for customer in CUSTOMERS
        for order in customer.orders
        for sku, _qty in order.items
        if sku not in catalogue
    }
    assert unknown == set()


def test_the_catalogue_covers_all_three_stock_states() -> None:
    """In stock, low and zero — so the agent has to report all three rather
    than learning that the answer is always yes."""
    quantities = [p.stock_qty for p in PRODUCTS]
    assert any(q == 0 for q in quantities), "nothing is out of stock"
    assert any(0 < q <= 6 for q in quantities), "nothing is low"
    assert any(q > 6 for q in quantities), "nothing is well stocked"


def test_every_demo_number_is_a_valid_phone_number() -> None:
    """The dataset must satisfy the same validator the agent writes through.

    It did not: the numbers were +1555010x, which libphonenumber rejects — a
    +1 number needs ten digits after the country code and those have seven. So
    a caller could be *identified* by a demo number (lookup is lenient) but
    could not dictate one to save_caller_details, which refuses what it cannot
    validate. You could not demo a number change against the demo data.

    +1 202 555 01xx is the block reserved for fiction that is still well
    formed, so these stay obviously fake without being invalid.
    """
    import phonenumbers

    for customer in CUSTOMERS:
        parsed = phonenumbers.parse(customer.phone, None)
        assert phonenumbers.is_valid_number(parsed), f"{customer.phone} is not valid"
