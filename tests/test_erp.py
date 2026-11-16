import pytest

from app.errors import NotFoundError
from app.services import erp


async def test_finds_product_by_partial_name(seed_products: None) -> None:
    # Callers say "the desk lamp", never the SKU.
    item = await erp.find_product("desk lamp")
    assert item.sku == "SKU-1"
    assert item.in_stock is True


async def test_finds_product_by_sku_case_insensitively(seed_products: None) -> None:
    assert (await erp.find_product("sku-1")).name == "Aeron Desk Lamp"


async def test_exact_sku_beats_a_name_substring(seed_products: None) -> None:
    """'Desk' matches both rows; the SKU match must win, not alphabetical order."""
    assert (await erp.find_product("SKU-2")).sku == "SKU-2"


async def test_out_of_stock_is_a_result_not_an_error(seed_products: None) -> None:
    item = await erp.find_product("Standing Desk Mat")
    assert item.in_stock is False
    assert item.stock_qty == 0


async def test_price_is_formatted_for_display(seed_products: None) -> None:
    assert (await erp.find_product("SKU-1")).price_display == "$129.00"


async def test_unknown_product_raises_with_a_caller_safe_message(
    seed_products: None,
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        await erp.find_product("hovercraft")
    # The message is read aloud, so it must not leak the query or internals.
    assert excinfo.value.caller_message == "I couldn't find that product in our catalogue."


async def test_order_lookup_is_case_insensitive(seed_orders: None) -> None:
    assert (await erp.get_order("ord-1")).status == "shipped"


async def test_unknown_order_raises(seed_orders: None) -> None:
    with pytest.raises(NotFoundError):
        await erp.get_order("ORD-NOPE")


async def test_orders_for_phone_returns_all_matches(seed_orders: None) -> None:
    orders = await erp.find_orders_for_phone("+15551234567")
    assert {o.order_ref for o in orders} == {"ORD-1", "ORD-2"}


async def test_orders_for_unknown_phone_is_empty_not_an_error(
    seed_orders: None,
) -> None:
    assert await erp.find_orders_for_phone("+10000000000") == []
