"""HubSpot adapter tests.

respx intercepts at the httpx transport layer, so the client's real retry,
timeout and error-mapping code runs — only the socket is fake.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.errors import IntegrationError, RateLimitedError
from app.services.hubspot import HubSpotClient, phone_variants

BASE = "https://api.hubapi.com"
SEARCH = f"{BASE}/crm/v3/objects/contacts/search"


@pytest.fixture
def client() -> HubSpotClient:
    return HubSpotClient(
        client=httpx.AsyncClient(base_url=BASE, headers={"Authorization": "Bearer test"})
    )


def _contact_payload() -> dict[str, object]:
    return {
        "results": [
            {
                "id": "101",
                "properties": {
                    "firstname": "Ada",
                    "lastname": "Lovelace",
                    "email": "ada@example.com",
                    "phone": "+15551234567",
                },
            }
        ]
    }


# --- phone normalisation ---------------------------------------------------


def test_phone_variants_cover_common_spellings() -> None:
    variants = phone_variants("+1 (555) 123-4567")
    assert "+15551234567" in variants
    assert "15551234567" in variants
    # US numbers are often stored without the country code.
    assert "5551234567" in variants


def test_phone_variants_are_deduplicated() -> None:
    assert len(phone_variants("+15551234567")) == len(set(phone_variants("+15551234567")))


# --- contact lookup --------------------------------------------------------


@respx.mock
async def test_finds_contact(client: HubSpotClient) -> None:
    respx.post(SEARCH).mock(return_value=httpx.Response(200, json=_contact_payload()))

    contact = await client.find_contact_by_phone("+15551234567")

    assert contact is not None
    assert contact.id == "101"
    assert contact.display_name == "Ada Lovelace"


@respx.mock
async def test_unknown_caller_returns_none_rather_than_raising(
    client: HubSpotClient,
) -> None:
    """An unmatched number is ordinary traffic, not a failure."""
    respx.post(SEARCH).mock(return_value=httpx.Response(200, json={"results": []}))

    assert await client.find_contact_by_phone("+15550000000") is None


@respx.mock
async def test_falls_back_to_alternative_phone_spellings(
    client: HubSpotClient,
) -> None:
    route = respx.post(SEARCH).mock(
        side_effect=[
            httpx.Response(200, json={"results": []}),  # +15551234567 — miss
            httpx.Response(200, json=_contact_payload()),  # 15551234567 — hit
        ]
    )

    contact = await client.find_contact_by_phone("+15551234567")

    assert contact is not None
    assert route.call_count == 2


@respx.mock
async def test_display_name_falls_back_when_no_name_recorded(
    client: HubSpotClient,
) -> None:
    respx.post(SEARCH).mock(
        return_value=httpx.Response(
            200, json={"results": [{"id": "7", "properties": {"phone": "+1555"}}]}
        )
    )
    contact = await client.find_contact_by_phone("+1555")
    assert contact is not None
    assert contact.display_name == "there"


# --- retry behaviour -------------------------------------------------------


@respx.mock
async def test_retries_transient_server_errors_then_succeeds(
    client: HubSpotClient,
) -> None:
    route = respx.post(SEARCH).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=_contact_payload()),
        ]
    )

    contact = await client.find_contact_by_phone("+15551234567")

    assert contact is not None
    assert route.call_count == 2


@respx.mock
async def test_gives_up_on_persistent_server_errors(client: HubSpotClient) -> None:
    route = respx.post(SEARCH).mock(return_value=httpx.Response(500))

    with pytest.raises(IntegrationError) as excinfo:
        await client.find_contact_by_phone("+15551234567")

    assert route.call_count == 3
    assert excinfo.value.retryable is True
    # The caller must never hear the upstream status code.
    assert "500" not in excinfo.value.caller_message


@respx.mock
async def test_does_not_retry_client_errors(client: HubSpotClient) -> None:
    """A 400 is our bug. Retrying just burns time while someone waits."""
    route = respx.post(SEARCH).mock(return_value=httpx.Response(400, text="bad scope"))

    with pytest.raises(IntegrationError):
        await client.find_contact_by_phone("+15551234567")

    assert route.call_count == 1


@respx.mock
async def test_rate_limiting_raises_after_exhausting_attempts(
    client: HubSpotClient,
) -> None:
    respx.post(SEARCH).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "0"})
    )

    with pytest.raises(RateLimitedError) as excinfo:
        await client.find_contact_by_phone("+15551234567")

    assert excinfo.value.retry_after == 0


@respx.mock
async def test_timeouts_are_retried_then_reported(client: HubSpotClient) -> None:
    route = respx.post(SEARCH).mock(side_effect=httpx.ConnectTimeout("timed out"))

    with pytest.raises(IntegrationError):
        await client.find_contact_by_phone("+15551234567")

    assert route.call_count == 3


# --- writes ----------------------------------------------------------------


@respx.mock
async def test_creates_contact_omitting_blank_fields(client: HubSpotClient) -> None:
    route = respx.post(f"{BASE}/crm/v3/objects/contacts").mock(
        return_value=httpx.Response(
            201, json={"id": "202", "properties": {"firstname": "Grace"}}
        )
    )

    contact = await client.create_contact(phone="+15559876543", first_name="Grace")

    assert contact.id == "202"
    sent = route.calls.last.request.read().decode()
    assert "lastname" not in sent, "empty properties should not be sent"


@respx.mock
async def test_logs_note_with_contact_association(client: HubSpotClient) -> None:
    route = respx.post(f"{BASE}/crm/v3/objects/notes").mock(
        return_value=httpx.Response(201, json={"id": "n-1"})
    )

    note_id = await client.log_call_note(contact_id="101", body="Summary text")

    assert note_id == "n-1"
    body = route.calls.last.request.read().decode()
    # Without the association the note is orphaned and never appears on the
    # contact's timeline — the entire point of writing it.
    assert '"associationTypeId": 202' in body or '"associationTypeId":202' in body
    assert "hs_timestamp" in body


# --- seeding helpers -------------------------------------------------------


@respx.mock
async def test_upsert_returns_existing_contact_without_creating(
    client: HubSpotClient,
) -> None:
    """Seeding writes to a real portal, so it must not pile up duplicates."""
    respx.post(SEARCH).mock(return_value=httpx.Response(200, json=_contact_payload()))
    create = respx.post(f"{BASE}/crm/v3/objects/contacts").mock(
        return_value=httpx.Response(201, json={"id": "999"})
    )

    contact, created = await client.upsert_contact(phone="+15551234567")

    assert contact.id == "101"
    assert created is False
    assert not create.called, "an existing contact must not be recreated"


@respx.mock
async def test_upsert_creates_when_the_number_is_unknown(
    client: HubSpotClient,
) -> None:
    respx.post(SEARCH).mock(return_value=httpx.Response(200, json={"results": []}))
    respx.post(f"{BASE}/crm/v3/objects/contacts").mock(
        return_value=httpx.Response(201, json={"id": "202", "properties": {"firstname": "Ada"}})
    )

    contact, created = await client.upsert_contact(phone="+15550100", first_name="Ada")

    assert created is True
    assert contact.id == "202"


@respx.mock
async def test_deal_stage_is_read_from_the_portal_not_hardcoded(
    client: HubSpotClient,
) -> None:
    """Stage ids differ per portal, so a hardcoded label would 400 elsewhere."""
    respx.get(f"{BASE}/crm/v3/pipelines/deals").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "stages": [
                            {"id": "stage-b", "displayOrder": 1},
                            {"id": "stage-a", "displayOrder": 0},
                        ]
                    }
                ]
            },
        )
    )

    assert await client.first_deal_stage() == "stage-a"


@respx.mock
async def test_no_pipelines_returns_none_rather_than_raising(
    client: HubSpotClient,
) -> None:
    respx.get(f"{BASE}/crm/v3/pipelines/deals").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    assert await client.first_deal_stage() is None


#: A pipeline shaped like HubSpot's: the outcome lives in stage metadata, and
#: the won stage is not the last one — "closed lost" usually is.
PIPELINE = {
    "results": [
        {
            "stages": [
                {
                    "id": "s-new",
                    "displayOrder": 0,
                    "metadata": {"isClosed": "false", "probability": "0.2"},
                },
                {
                    "id": "s-won",
                    "displayOrder": 1,
                    "metadata": {"isClosed": "true", "probability": "1.0"},
                },
                {
                    "id": "s-lost",
                    "displayOrder": 2,
                    "metadata": {"isClosed": "true", "probability": "0.0"},
                },
            ]
        }
    ]
}


@respx.mock
async def test_won_stage_is_read_from_metadata_not_from_position(
    client: HubSpotClient,
) -> None:
    """Both won and lost are "closed"; only probability separates them, and
    taking the last stage would mark every sale lost."""
    respx.get(f"{BASE}/crm/v3/pipelines/deals").mock(
        return_value=httpx.Response(200, json=PIPELINE)
    )
    assert await client.closed_won_stage() == "s-won"


@respx.mock
async def test_closing_a_deal_moves_it_to_the_won_stage(
    client: HubSpotClient,
) -> None:
    respx.get(f"{BASE}/crm/v3/pipelines/deals").mock(
        return_value=httpx.Response(200, json=PIPELINE)
    )
    patch = respx.patch(f"{BASE}/crm/v3/objects/deals/d-9").mock(
        return_value=httpx.Response(200, json={"id": "d-9"})
    )

    assert await client.close_deal_won("d-9") is True
    sent = json.loads(patch.calls[0].request.read())
    assert sent == {"properties": {"dealstage": "s-won"}}


@respx.mock
async def test_the_won_stage_is_resolved_once_not_per_deal(
    client: HubSpotClient,
) -> None:
    """It is the same for every deal, and the lookup is a round trip while a
    caller waits."""
    pipelines = respx.get(f"{BASE}/crm/v3/pipelines/deals").mock(
        return_value=httpx.Response(200, json=PIPELINE)
    )
    respx.patch(url__regex=rf"{BASE}/crm/v3/objects/deals/.*").mock(
        return_value=httpx.Response(200, json={})
    )

    await client.close_deal_won("d-1")
    await client.close_deal_won("d-2")
    assert pipelines.call_count == 1


@respx.mock
async def test_a_portal_with_no_won_stage_reports_it_rather_than_guessing(
    client: HubSpotClient,
) -> None:
    """Patching a made-up stage id succeeds and leaves the deal nowhere."""
    respx.get(f"{BASE}/crm/v3/pipelines/deals").mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"stages": [{"id": "s-only", "metadata": {}}]}]},
        )
    )
    patch = respx.patch(url__regex=rf"{BASE}/crm/v3/objects/deals/.*").mock(
        return_value=httpx.Response(200, json={})
    )

    assert await client.close_deal_won("d-1") is False
    assert not patch.called


@respx.mock
async def test_deal_is_associated_with_its_contact(client: HubSpotClient) -> None:
    """An unassociated deal is orphaned and invisible on the contact record.

    The association goes through the v4 default endpoint, which needs no
    associationTypeId — those are easy to get wrong and a wrong one fails
    silently rather than erroring.
    """
    create = respx.post(f"{BASE}/crm/v3/objects/deals").mock(
        return_value=httpx.Response(201, json={"id": "d-1"})
    )
    associate = respx.put(
        f"{BASE}/crm/v4/objects/deals/d-1/associations/default/contacts/101"
    ).mock(return_value=httpx.Response(200, json={}))

    deal_id = await client.create_deal(
        name="Ada — ORD-5002", amount_cents=74900, stage="s1", contact_id="101"
    )

    assert deal_id == "d-1"
    assert associate.called, "deal must be associated with the contact"
    body = create.calls.last.request.read().decode()
    assert '"amount": "749.00"' in body or '"amount":"749.00"' in body


@respx.mock
async def test_task_is_associated_with_its_contact(client: HubSpotClient) -> None:
    respx.post(f"{BASE}/crm/v3/objects/tasks").mock(
        return_value=httpx.Response(201, json={"id": "t-1"})
    )
    associate = respx.put(
        f"{BASE}/crm/v4/objects/tasks/t-1/associations/default/contacts/101"
    ).mock(return_value=httpx.Response(200, json={}))

    await client.create_task(subject="Call back Ada", body="…", contact_id="101")

    assert associate.called


@respx.mock
async def test_update_contact_skips_the_call_when_nothing_was_given(
    client: HubSpotClient,
) -> None:
    """A caller who gives no new detail should not cost a network round trip."""
    route = respx.patch(f"{BASE}/crm/v3/objects/contacts/101").mock(
        return_value=httpx.Response(200, json={})
    )

    await client.update_contact(contact_id="101")

    assert not route.called


# --- spoken input ----------------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("ada at example dot com", "ada@example.com"),
        ("ada dot okonkwo at example dot com", "ada.okonkwo@example.com"),
        ("Ada At Example Dot Com", "ada@example.com"),
        ("ada_test at example dot com", "ada_test@example.com"),
        ("ada @ example.com", "ada@example.com"),
        ("already@valid.com", "already@valid.com"),
    ],
)
def test_dictated_emails_are_repaired(spoken: str, expected: str) -> None:
    """Speech recognition transcribes a spoken address literally.

    Sent as-is, HubSpot rejects the whole record with INVALID_EMAIL and the
    contact is never created — losing the name and number too.
    """
    from app.services.hubspot import normalise_spoken_email

    assert normalise_spoken_email(spoken) == expected


@pytest.mark.parametrize("junk", ["not an email", "", "   ", "ada at example"])
def test_unrepairable_emails_are_dropped_not_sent(junk: str) -> None:
    """A contact with no email is useful; a failed tool call is not."""
    from app.services.hubspot import normalise_spoken_email

    assert normalise_spoken_email(junk) is None


@respx.mock
async def test_dictated_email_does_not_reach_hubspot_raw(
    client: HubSpotClient,
) -> None:
    route = respx.post(f"{BASE}/crm/v3/objects/contacts").mock(
        return_value=httpx.Response(201, json={"id": "1", "properties": {}})
    )

    await client.create_contact(
        phone="+15550100", first_name="Ada", email="ada at example dot com"
    )

    body = route.calls.last.request.read().decode()
    assert "ada@example.com" in body
    assert "dot com" not in body


@respx.mock
async def test_unusable_email_is_omitted_so_the_contact_still_saves(
    client: HubSpotClient,
) -> None:
    route = respx.post(f"{BASE}/crm/v3/objects/contacts").mock(
        return_value=httpx.Response(201, json={"id": "1", "properties": {}})
    )

    await client.create_contact(
        phone="+15550100", first_name="Ada", email="complete nonsense"
    )

    body = route.calls.last.request.read().decode()
    assert "email" not in body
    assert "Ada" in body, "the rest of the record must still be written"


@respx.mock
async def test_contacts_are_read_in_one_batch_not_one_by_one(
    client: HubSpotClient,
) -> None:
    """The dashboard needs current details for a page of calls. A request per
    row would make it the slowest page in the app and spend rate limit in
    proportion to traffic."""
    batch = respx.post(f"{BASE}/crm/v3/objects/contacts/batch/read").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"id": "1", "properties": {"firstname": "Ada", "phone": "+1"}},
                    {"id": "2", "properties": {"firstname": "Grace", "phone": "+2"}},
                ]
            },
        )
    )

    found = await client.batch_read_contacts(["1", "2"])

    assert batch.call_count == 1
    assert set(found) == {"1", "2"}
    assert found["2"].phone == "+2"


@respx.mock
async def test_a_contact_that_no_longer_exists_is_simply_absent(
    client: HubSpotClient,
) -> None:
    """Deleted and merged records are ordinary. HubSpot answers 207 with the
    survivors, and a missing row must not fail the whole page."""
    respx.post(f"{BASE}/crm/v3/objects/contacts/batch/read").mock(
        return_value=httpx.Response(
            207,
            json={"results": [{"id": "1", "properties": {"firstname": "Ada"}}]},
        )
    )

    found = await client.batch_read_contacts(["1", "gone"])

    assert set(found) == {"1"}


async def test_reading_no_contacts_makes_no_request(client: HubSpotClient) -> None:
    assert await client.batch_read_contacts([]) == {}
