"""HubSpot CRM adapter.

The only module that knows HubSpot's wire format. Everything above it deals in
the dataclasses defined here, so swapping HubSpot for another CRM means
rewriting this file and nothing else.

Two behaviours worth knowing about:

*Retries.* HubSpot rate-limits aggressively (429) and has routine 5xx blips.
Both are retried with backoff, honouring ``Retry-After`` when present. A call
is in progress while this runs, so the budget is deliberately small — better to
apologise after two seconds than to leave the caller in silence for ten.

*Phone matching.* Caller ID arrives from SIP in E.164 (``+15551234567``) but
HubSpot records are hand-entered in whatever format the rep typed. We search
several normalisations rather than assuming the data is clean; an unmatched
number is a normal outcome, not an error.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import phonenumbers

from app.config import settings
from app.errors import ConfigurationError, IntegrationError, RateLimitedError
from app.observability.logging import get_logger

log = get_logger(__name__)

# Kept small on purpose: this runs while someone is holding a phone to their ear.
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.4
REQUEST_TIMEOUT_SECONDS = 5.0

# HubSpot's built-in association type id for note -> contact.
NOTE_TO_CONTACT_ASSOCIATION_TYPE_ID = 202


@dataclass(frozen=True)
class Contact:
    id: str
    first_name: str | None
    last_name: str | None
    email: str | None
    phone: str | None

    @property
    def display_name(self) -> str:
        parts = [p for p in (self.first_name, self.last_name) if p]
        return " ".join(parts) if parts else "there"


#: Good enough to reject what speech recognition mangles, without trying to
#: be RFC 5322 — the authority on whether an address is real is HubSpot.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

#: Spoken forms, longest first so "dot com" is handled before "dot".
_SPOKEN_EMAIL = [
    (re.compile(r"\s+at\s+", re.I), "@"),
    (re.compile(r"\s+dot\s+", re.I), "."),
    (re.compile(r"\s+underscore\s+", re.I), "_"),
    (re.compile(r"\s+dash\s+|\s+hyphen\s+", re.I), "-"),
]


def normalise_spoken_email(raw: str) -> str | None:
    """Turn a dictated email into a real one, or None if it cannot be.

    A caller says "ada at example dot com" and speech recognition transcribes
    exactly that. Sent as-is, HubSpot rejects the whole record with a 400 and
    the contact is never created — so the address is repaired here, and
    dropped entirely if it still is not valid.

    Returning None is deliberate: a contact with no email is useful, a failed
    tool call is not.
    """
    if not raw:
        return None
    candidate = raw.strip()
    for pattern, replacement in _SPOKEN_EMAIL:
        candidate = pattern.sub(replacement, candidate)
    # Speech often leaves stray spaces around the parts it did transcribe.
    candidate = candidate.replace(" ", "").lower()
    return candidate if _EMAIL_RE.match(candidate) else None


def normalise_spoken_phone(raw: str) -> str | None:
    """Turn a dictated number into E.164, or None if it cannot be.

    Speech recognition renders numbers as digits, words, or a mix — "plus nine
    nine eight" and "+998" both occur. Anything that is not plausibly a phone
    number is dropped rather than written, for the same reason as emails: a
    bad value fails the whole update.
    """
    if not raw:
        return None
    spoken_digits = {
        "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    }
    text = raw.lower().replace("plus", "+")
    for word, digit in spoken_digits.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    digits = re.sub(r"[^\d+]", "", text)
    digits = "+" + digits.lstrip("+") if digits.startswith("+") else digits
    bare = digits.lstrip("+")
    if not bare.isdigit():
        return None
    candidate = digits if digits.startswith("+") else f"+{digits}"

    # A length check is not enough, and the gap is not theoretical: a misheard
    # digit produced "+999899999", which is nine digits and passes any range
    # test, but +999 is not an assigned country code. It was accepted, written
    # to the CRM, and the caller's orders were relinked to a number nobody can
    # ring from — the exact failure the relink exists to prevent.
    #
    # libphonenumber knows the assigned prefixes and the valid lengths within
    # each country, which no rule written here would.
    try:
        parsed = phonenumbers.parse(candidate, None)
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_valid_number(parsed):
        return None
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def phone_variants(raw: str) -> list[str]:
    """Plausible spellings of a number, most specific first.

    ``+1 (555) 123-4567`` may be stored as ``+15551234567``, ``5551234567`` or
    ``555-123-4567``. HubSpot search is exact-match, so we try each.
    """
    digits = re.sub(r"\D", "", raw)
    variants = [raw, f"+{digits}", digits]
    # US numbers are frequently stored without the country code.
    if len(digits) == 11 and digits.startswith("1"):
        variants.append(digits[1:])
    seen: set[str] = set()
    unique: list[str] = []
    for variant in variants:
        if variant and variant not in seen:
            seen.add(variant)
            unique.append(variant)
    return unique


class HubSpotClient:
    """Async HubSpot CRM v3 client.

    Construct once per process and reuse — the underlying connection pool is
    what keeps per-request latency low enough for a live call.
    """

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        if not settings.hubspot_access_token and client is None:
            raise ConfigurationError(
                "HUBSPOT_ACCESS_TOKEN is not set; the agent cannot reach the CRM"
            )
        self._client = client or httpx.AsyncClient(
            base_url=settings.hubspot_base_url,
            headers={
                "Authorization": f"Bearer {settings.hubspot_access_token}",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        self._cached_stage: str | None = None
        self._cached_won_stage: str | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- transport ---------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        last_exc: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as exc:
                last_exc = exc
                log.warning(
                    "hubspot_timeout", method=method, path=path, attempt=attempt
                )
            except httpx.HTTPError as exc:
                last_exc = exc
                log.warning(
                    "hubspot_transport_error",
                    method=method,
                    path=path,
                    attempt=attempt,
                    error=str(exc),
                )
            else:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                log.info(
                    "hubspot_request",
                    method=method,
                    path=path,
                    status=response.status_code,
                    elapsed_ms=elapsed_ms,
                    attempt=attempt,
                )

                if response.status_code == 429:
                    retry_after = _retry_after_seconds(response)
                    if attempt == MAX_ATTEMPTS:
                        raise RateLimitedError(
                            "HubSpot rate limit exhausted",
                            retry_after=retry_after,
                            path=path,
                        )
                    await asyncio.sleep(retry_after or _backoff(attempt))
                    continue

                if response.status_code >= 500:
                    if attempt == MAX_ATTEMPTS:
                        raise IntegrationError(
                            f"HubSpot returned {response.status_code}",
                            status=response.status_code,
                            path=path,
                        )
                    await asyncio.sleep(_backoff(attempt))
                    continue

                if response.status_code >= 400:
                    # 4xx is our bug (bad payload, bad scope), not a blip.
                    # Retrying would just burn call time.
                    raise IntegrationError(
                        f"HubSpot rejected the request: {response.status_code}",
                        status=response.status_code,
                        path=path,
                        body=response.text[:500],
                    )

                if not response.content:
                    return {}
                return response.json()  # type: ignore[no-any-return]

            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(_backoff(attempt))

        raise IntegrationError(
            "HubSpot unreachable after retries", path=path, cause=str(last_exc)
        )

    # --- contacts ----------------------------------------------------------

    async def find_contact_by_phone(self, phone: str) -> Contact | None:
        """Return the first contact matching any spelling of ``phone``.

        ``None`` means "not a known caller", which is an ordinary outcome — the
        agent offers to create a record rather than treating it as a failure.
        """
        for variant in phone_variants(phone):
            payload = {
                "filterGroups": [
                    {"filters": [{"propertyName": prop, "operator": "EQ", "value": variant}]}
                    for prop in ("phone", "mobilephone")
                ],
                "properties": ["firstname", "lastname", "email", "phone"],
                "limit": 1,
            }
            data = await self._request(
                "POST", "/crm/v3/objects/contacts/search", json=payload
            )
            results = data.get("results") or []
            if results:
                log.info("hubspot_contact_matched", variant=variant)
                return _to_contact(results[0])

        log.info("hubspot_contact_not_found", phone=phone)
        return None

    async def batch_read_contacts(self, contact_ids: list[str]) -> dict[str, Contact]:
        """Current details for many contacts, keyed by id, in one request.

        The dashboard shows the number that rang — a snapshot of the call —
        next to the number the contact has *now*, which has to come from the
        CRM because that is where it lives. One request per page rather than
        one per row: a request per row would make the call list the slowest
        page in the app and would spend rate limit proportional to traffic.

        Missing ids are simply absent from the result. Deleted contacts and
        merged records are ordinary here, not errors.
        """
        if not contact_ids:
            return {}
        found: dict[str, Contact] = {}
        # HubSpot caps a batch read at 100 inputs.
        for start in range(0, len(contact_ids), 100):
            chunk = contact_ids[start : start + 100]
            data = await self._request(
                "POST",
                "/crm/v3/objects/contacts/batch/read",
                json={
                    "properties": ["firstname", "lastname", "email", "phone"],
                    "inputs": [{"id": contact_id} for contact_id in chunk],
                },
            )
            for result in data.get("results") or []:
                contact = _to_contact(result)
                found[contact.id] = contact
        log.info("hubspot_contacts_read", asked=len(contact_ids), found=len(found))
        return found

    async def create_contact(
        self,
        *,
        phone: str,
        first_name: str | None = None,
        last_name: str | None = None,
        email: str | None = None,
    ) -> Contact:
        properties = {
            "phone": phone,
            "firstname": first_name,
            "lastname": last_name,
            # Dropped rather than sent raw: an unusable address fails the whole
            # create, losing the name and number too.
            "email": normalise_spoken_email(email) if email else None,
        }
        data = await self._request(
            "POST",
            "/crm/v3/objects/contacts",
            json={"properties": {k: v for k, v in properties.items() if v}},
        )
        contact = _to_contact(data)
        log.info("hubspot_contact_created", contact_id=contact.id)
        return contact

    # --- activity logging --------------------------------------------------

    async def log_call_note(
        self,
        *,
        contact_id: str,
        body: str,
        occurred_at: datetime | None = None,
    ) -> str:
        """Attach a note to the contact's timeline and return its id.

        This is what makes the call visible to a human in HubSpot afterwards.
        """
        timestamp = occurred_at or datetime.now(UTC)
        payload = {
            "properties": {
                "hs_note_body": body,
                # HubSpot wants epoch milliseconds.
                "hs_timestamp": int(timestamp.timestamp() * 1000),
            },
            "associations": [
                {
                    "to": {"id": contact_id},
                    "types": [
                        {
                            "associationCategory": "HUBSPOT_DEFINED",
                            "associationTypeId": NOTE_TO_CONTACT_ASSOCIATION_TYPE_ID,
                        }
                    ],
                }
            ],
        }
        data = await self._request("POST", "/crm/v3/objects/notes", json=payload)
        note_id = str(data.get("id", ""))
        log.info("hubspot_note_logged", contact_id=contact_id, note_id=note_id)
        return note_id

    async def upsert_contact(
        self,
        *,
        phone: str,
        first_name: str | None = None,
        last_name: str | None = None,
        email: str | None = None,
        company: str | None = None,
    ) -> tuple[Contact, bool]:
        """Find a contact by phone or create it. Returns (contact, created).

        Used by seeding, which must be safe to re-run without piling up
        duplicate records in a real portal.
        """
        existing = await self.find_contact_by_phone(phone)
        if existing is not None:
            return existing, False

        properties = {
            "phone": phone,
            "firstname": first_name,
            "lastname": last_name,
            "email": normalise_spoken_email(email) if email else None,
            "company": company,
        }
        data = await self._request(
            "POST",
            "/crm/v3/objects/contacts",
            json={"properties": {k: v for k, v in properties.items() if v}},
        )
        contact = _to_contact(data)
        log.info("hubspot_contact_created", contact_id=contact.id)
        return contact, True

    async def update_contact(
        self,
        *,
        contact_id: str,
        email: str | None = None,
        company: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        phone: str | None = None,
    ) -> None:
        """Patch properties on an existing contact. Blank values are ignored.

        Changing ``phone`` is not like changing a name. The number is the join
        key between this CRM and the ERP: identification looks the caller up
        here, order lookup matches the same number in Postgres. Change it on
        one side only and the caller is greeted by name and then told they have
        no orders — see ``erp.relink_orders``.
        """
        properties = {
            "email": normalise_spoken_email(email) if email else None,
            "company": company,
            "firstname": first_name,
            "lastname": last_name,
            "phone": normalise_spoken_phone(phone) if phone else None,
        }
        supplied = {k: v for k, v in properties.items() if v}
        if not supplied:
            return
        await self._request(
            "PATCH",
            f"/crm/v3/objects/contacts/{contact_id}",
            json={"properties": supplied},
        )
        log.info("hubspot_contact_updated", contact_id=contact_id, fields=list(supplied))

    # --- associations ------------------------------------------------------

    async def associate(
        self,
        *,
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
    ) -> None:
        """Create the default association between two records.

        The v4 ``associations/default`` endpoint takes no body and no
        ``associationTypeId`` — those ids are easy to get wrong and a wrong one
        silently orphans the record rather than erroring, so preferring this
        endpoint removes a whole class of quiet failure.
        """
        await self._request(
            "PUT",
            f"/crm/v4/objects/{from_type}/{from_id}/associations/default/{to_type}/{to_id}",
        )
        log.info("hubspot_associated", from_type=from_type, to_type=to_type)

    # --- tasks -------------------------------------------------------------

    async def create_task(
        self,
        *,
        subject: str,
        body: str,
        contact_id: str | None = None,
        due_at: datetime | None = None,
        priority: str = "MEDIUM",
    ) -> str:
        """Create a follow-up task, optionally on a contact's record.

        This is how the agent hands work to a human: a caller who needs
        something the agent cannot do leaves a task in someone's queue rather
        than a note nobody is prompted to read.
        """
        due = due_at or (datetime.now(UTC) + timedelta(days=1))
        data = await self._request(
            "POST",
            "/crm/v3/objects/tasks",
            json={
                "properties": {
                    "hs_task_subject": subject,
                    "hs_task_body": body,
                    "hs_task_status": "NOT_STARTED",
                    "hs_task_priority": priority,
                    "hs_timestamp": int(due.timestamp() * 1000),
                }
            },
        )
        task_id = str(data.get("id", ""))
        if contact_id:
            await self.associate(
                from_type="tasks", from_id=task_id,
                to_type="contacts", to_id=contact_id,
            )
        log.info("hubspot_task_created", task_id=task_id, subject=subject)
        return task_id

    # --- deals -------------------------------------------------------------

    async def create_deal(
        self,
        *,
        name: str,
        amount_cents: int,
        stage: str | None = None,
        contact_id: str | None = None,
    ) -> str:
        """Create a deal, optionally associated with a contact.

        ``stage`` must be an internal stage id from the portal's own pipeline,
        not a display label — those differ per portal, which is why the seeder
        reads them from the API rather than hardcoding them.
        """
        # Stage ids are per-portal. Resolve once per process rather than
        # hardcoding a label, and never during the first deal of a call if it
        # can be avoided — it is an extra round trip while someone waits.
        if stage is None:
            if self._cached_stage is None:
                self._cached_stage = await self.first_deal_stage()
            stage = self._cached_stage

        properties: dict[str, Any] = {
            "dealname": name,
            "amount": f"{amount_cents / 100:.2f}",
        }
        if stage:
            properties["dealstage"] = stage

        data = await self._request(
            "POST", "/crm/v3/objects/deals", json={"properties": properties}
        )
        deal_id = str(data.get("id", ""))
        if contact_id:
            await self.associate(
                from_type="deals", from_id=deal_id,
                to_type="contacts", to_id=contact_id,
            )
        log.info("hubspot_deal_created", deal_id=deal_id, name=name)
        return deal_id

    async def find_deal_by_name(self, name: str) -> str | None:
        """Deal id for an exact name match, or None.

        Seeding uses this so re-running does not stack duplicate deals in a
        real portal — contacts dedupe by phone, but deals have no natural key.
        """
        data = await self._request(
            "POST",
            "/crm/v3/objects/deals/search",
            json={
                "filterGroups": [
                    {
                        "filters": [
                            {
                                "propertyName": "dealname",
                                "operator": "EQ",
                                "value": name,
                            }
                        ]
                    }
                ],
                "limit": 1,
            },
        )
        results = data.get("results") or []
        return str(results[0]["id"]) if results else None

    async def first_deal_stage(self) -> str | None:
        """The first stage of the portal's default deal pipeline.

        Stage ids are per-portal, so seeding discovers one instead of guessing
        at a label like "appointmentscheduled" that may not exist.
        """
        data = await self._request("GET", "/crm/v3/pipelines/deals")
        pipelines = data.get("results") or []
        if not pipelines:
            return None
        stages = sorted(
            pipelines[0].get("stages") or [],
            key=lambda s: s.get("displayOrder", 0),
        )
        return str(stages[0]["id"]) if stages else None

    async def closed_won_stage(self) -> str | None:
        """The won stage of the portal's default deal pipeline.

        Discovered rather than hardcoded, for the same reason as
        ``first_deal_stage``: ``closedwon`` is the id in a stock portal but not
        in a customised one, and a wrong stage id is accepted as a property
        value that then matches nothing in the pipeline.

        HubSpot marks the outcome in stage metadata — ``isClosed`` with
        ``probability`` 1 is won, ``probability`` 0 is lost — so read that
        rather than matching on a display label somebody may have renamed.
        """
        data = await self._request("GET", "/crm/v3/pipelines/deals")
        pipelines = data.get("results") or []
        if not pipelines:
            return None
        for stage in pipelines[0].get("stages") or []:
            metadata = stage.get("metadata") or {}
            closed = str(metadata.get("isClosed", "")).lower() == "true"
            if closed and float(metadata.get("probability", 0) or 0) >= 1:
                return str(stage["id"])
        return None

    async def close_deal_won(self, deal_id: str, *, order_ref: str | None = None) -> bool:
        """Mark a deal won, stamping the ERP order it became.

        False if this portal's pipeline has no won stage. Caches the stage id
        for the life of the client: it is the same for every deal, and
        resolving it costs a round trip while a caller waits.

        ``order_ref`` is written as a *pointer*, not a copy. Someone looking at
        the won deal in HubSpot needs to know which order fulfils it; they do
        not need its status mirrored here, because the ERP owns that and a
        mirror would go stale the moment the order ships. Stock never comes
        here at all — HubSpot has no model for it, and giving an inventory
        level a second home is how the two stop agreeing.
        """
        if self._cached_won_stage is None:
            self._cached_won_stage = await self.closed_won_stage()
        if not self._cached_won_stage:
            log.warning("hubspot_no_won_stage", deal_id=deal_id)
            return False

        properties: dict[str, Any] = {"dealstage": self._cached_won_stage}
        if order_ref:
            properties["description"] = f"Fulfilled by ERP order {order_ref}."
        await self._request(
            "PATCH",
            f"/crm/v3/objects/deals/{deal_id}",
            json={"properties": properties},
        )
        log.info("hubspot_deal_won", deal_id=deal_id, order_ref=order_ref)
        return True

    async def update_deal_stage(self, *, deal_id: str, stage: str) -> None:
        await self._request(
            "PATCH",
            f"/crm/v3/objects/deals/{deal_id}",
            json={"properties": {"dealstage": stage}},
        )
        log.info("hubspot_deal_stage_updated", deal_id=deal_id, stage=stage)


def _backoff(attempt: int) -> float:
    # 2.0 rather than 2: `int ** int` is Any to a type checker, because a
    # negative exponent would make it a float.
    return BACKOFF_BASE_SECONDS * (2.0 ** (attempt - 1))


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _to_contact(payload: dict[str, Any]) -> Contact:
    properties = payload.get("properties") or {}
    return Contact(
        id=str(payload.get("id", "")),
        first_name=properties.get("firstname"),
        last_name=properties.get("lastname"),
        email=properties.get("email"),
        phone=properties.get("phone"),
    )
