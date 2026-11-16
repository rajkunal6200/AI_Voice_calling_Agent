import asyncio
import re
from datetime import UTC, datetime, timedelta

import pytest

from app.errors import UserFacingError, ValidationError
from app.services import scheduling


def _tomorrow_at(hour: int) -> datetime:
    return (datetime.now(UTC) + timedelta(days=1)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )


async def test_books_a_free_slot() -> None:
    when = _tomorrow_at(10)
    appointment = await scheduling.book(starts_at=when, call_id="call-1")
    assert appointment.starts_at == when
    assert appointment.duration_minutes == scheduling.SLOT_MINUTES


async def test_concurrent_bookings_of_one_slot_produce_exactly_one_winner() -> None:
    """The load-bearing concurrency guarantee.

    Two callers offered the same slot both accept. A read-then-write would
    double-book; the UNIQUE constraint means one INSERT loses instead.
    """
    when = _tomorrow_at(11)

    results = await asyncio.gather(
        scheduling.book(starts_at=when, call_id="call-A"),
        scheduling.book(starts_at=when, call_id="call-B"),
        return_exceptions=True,
    )

    booked = [r for r in results if isinstance(r, scheduling.Appointment)]
    rejected = [r for r in results if isinstance(r, UserFacingError)]
    assert len(booked) == 1, "exactly one caller should get the slot"
    assert len(rejected) == 1, "the other must be told, not silently double-booked"
    assert "just taken" in rejected[0].caller_message


async def test_booking_in_the_past_is_rejected() -> None:
    with pytest.raises(ValidationError):
        await scheduling.book(starts_at=datetime.now(UTC) - timedelta(hours=1))


async def test_naive_datetime_is_rejected() -> None:
    """A naive datetime would be stored as if it were UTC and silently shift."""
    with pytest.raises(ValidationError):
        await scheduling.book(starts_at=datetime.now() + timedelta(days=1))  # noqa: DTZ005


async def test_offered_slots_exclude_already_booked_times() -> None:
    slots = await scheduling.available_slots(limit=3)
    assert slots, "expected some free slots to start with"
    taken = slots[0].starts_at

    await scheduling.book(starts_at=taken, call_id="call-1")

    remaining = await scheduling.available_slots(limit=3)
    assert taken not in [s.starts_at for s in remaining]


async def test_slots_are_within_business_hours_on_weekdays() -> None:
    for slot in await scheduling.available_slots(limit=6):
        assert slot.starts_at.hour in scheduling.BUSINESS_HOURS
        assert slot.starts_at.weekday() < 5


async def test_slot_reads_naturally_when_spoken() -> None:
    """The agent reads this aloud, so it must not be an ISO timestamp."""
    slot = scheduling.Slot(starts_at=_tomorrow_at(14))
    spoken = slot.spoken

    assert ":00 PM" in spoken
    # A weekday name, and no ISO date/separator anywhere in the string.
    assert any(day in spoken for day in ("Monday", "Tuesday", "Wednesday",
                                         "Thursday", "Friday", "Saturday", "Sunday"))
    assert not re.search(r"\d{4}-\d{2}-\d{2}", spoken)
    assert not re.search(r"\dT\d", spoken)
