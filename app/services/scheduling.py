"""Appointment booking.

The interesting constraint is concurrency. Two callers can be on the line at
the same time, both offered the same 2pm slot, and both accept. A
read-then-write ("is 2pm free? yes -> insert") double-books under that race.

So the free/busy read is advisory only, and the booking itself relies on the
UNIQUE constraint on ``appointments.starts_at``: the second INSERT loses, and
that loss is reported to the caller as "just gone" rather than silently
producing two bookings. Keep that shape — don't replace it with a SELECT.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.db import fetch_all, session_factory
from app.errors import UserFacingError, ValidationError
from app.observability.logging import get_logger

log = get_logger(__name__)

SLOT_MINUTES = 30
BUSINESS_HOURS = range(9, 17)  # 09:00–16:30, local to the business.
LOOKAHEAD_DAYS = 5


@dataclass(frozen=True)
class Slot:
    starts_at: datetime

    @property
    def spoken(self) -> str:
        """A form that sounds natural read aloud, not an ISO timestamp."""
        return self.starts_at.strftime("%A %-d %B at %-I:%M %p")


@dataclass(frozen=True)
class Appointment:
    id: int
    starts_at: datetime
    duration_minutes: int


async def available_slots(*, limit: int = 4) -> list[Slot]:
    """Next few free slots. Advisory — booking is what actually reserves."""
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    horizon = now + timedelta(days=LOOKAHEAD_DAYS)

    taken_rows = await fetch_all(
        "SELECT starts_at FROM appointments WHERE starts_at BETWEEN :start AND :end",
        start=now,
        end=horizon,
    )
    taken = {row["starts_at"] for row in taken_rows}

    slots: list[Slot] = []
    cursor = now
    while cursor < horizon and len(slots) < limit:
        if cursor.weekday() < 5 and cursor.hour in BUSINESS_HOURS and cursor not in taken:
            slots.append(Slot(starts_at=cursor))
        cursor += timedelta(minutes=SLOT_MINUTES)

    log.info("slots_offered", count=len(slots))
    return slots


async def book(
    *,
    starts_at: datetime,
    call_id: str | None = None,
    hubspot_contact_id: str | None = None,
    notes: str | None = None,
) -> Appointment:
    """Reserve a slot, or raise if someone else took it first."""
    if starts_at.tzinfo is None:
        raise ValidationError(
            "starts_at must be timezone-aware",
            caller_message="I didn't catch the time — could you say it again?",
        )
    if starts_at <= datetime.now(UTC):
        raise ValidationError(
            "cannot book in the past",
            caller_message="That time has already passed. Could you pick another?",
            starts_at=starts_at.isoformat(),
        )

    try:
        async with session_factory() as session:
            result = await session.execute(
                text(
                    """
                    INSERT INTO appointments
                        (call_id, hubspot_contact_id, starts_at, duration_minutes, notes)
                    VALUES
                        (:call_id, :contact_id, :starts_at, :duration, :notes)
                    RETURNING id, starts_at, duration_minutes
                    """
                ),
                {
                    "call_id": call_id,
                    "contact_id": hubspot_contact_id,
                    "starts_at": starts_at,
                    "duration": SLOT_MINUTES,
                    "notes": notes,
                },
            )
            row = result.mappings().one()
            await session.commit()
    except IntegrityError:
        # The UNIQUE constraint fired: another call booked this slot between
        # us offering it and the caller accepting.
        log.info("slot_already_taken", starts_at=starts_at.isoformat())
        raise UserFacingError(
            f"slot {starts_at.isoformat()} already booked",
            caller_message=(
                "Sorry, that slot was just taken. Let me offer you another time."
            ),
            starts_at=starts_at.isoformat(),
        ) from None

    appointment = Appointment(
        id=row["id"],
        starts_at=row["starts_at"],
        duration_minutes=row["duration_minutes"],
    )
    log.info(
        "appointment_booked",
        appointment_id=appointment.id,
        starts_at=appointment.starts_at.isoformat(),
        contact_id=hubspot_contact_id,
    )
    return appointment
