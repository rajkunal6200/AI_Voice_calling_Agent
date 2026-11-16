import httpx
import pytest
from httpx import ASGITransport

from app.main import app
from app.services import calls


@pytest.fixture
async def client() -> httpx.AsyncClient:
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest.fixture
async def a_finished_call() -> str:
    call_id = "api-test-1"
    await calls.start_call(call_id=call_id, direction="inbound", from_number="+1555")
    await calls.record_event(
        call_id=call_id, kind="agent_speech", payload={"text": "Hello"}
    )
    await calls.record_event(
        call_id=call_id,
        kind="error",
        name="check_stock",
        payload={"type": "IntegrationError", "reason": "upstream down"},
        latency_ms=1200,
    )
    await calls.end_call(call_id=call_id, status="failed", summary="Upstream failed.")
    return call_id


async def test_health(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_lists_calls_newest_first(client: httpx.AsyncClient) -> None:
    await calls.start_call(call_id="older", direction="inbound")
    await calls.start_call(call_id="newer", direction="outbound")

    body = (await client.get("/api/calls")).json()

    assert [c["call_id"] for c in body] == ["newer", "older"]


async def test_get_call_returns_its_summary(
    client: httpx.AsyncClient, a_finished_call: str
) -> None:
    body = (await client.get(f"/api/calls/{a_finished_call}")).json()
    assert body["status"] == "failed"
    assert body["summary"] == "Upstream failed."


async def test_unknown_call_is_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/calls/does-not-exist")).status_code == 404


async def test_events_for_unknown_call_are_404_not_an_empty_list(
    client: httpx.AsyncClient,
) -> None:
    """The UI must distinguish "no such call" from "nothing recorded yet"."""
    assert (await client.get("/api/calls/nope/events")).status_code == 404


async def test_events_are_returned_in_order(
    client: httpx.AsyncClient, a_finished_call: str
) -> None:
    body = (await client.get(f"/api/calls/{a_finished_call}/events")).json()
    assert [e["kind"] for e in body] == ["agent_speech", "error"]


async def test_error_feed_surfaces_the_failure(
    client: httpx.AsyncClient, a_finished_call: str
) -> None:
    body = (await client.get("/api/errors")).json()
    assert len(body) == 1
    assert body[0]["call_id"] == a_finished_call
    assert body[0]["payload"]["reason"] == "upstream down"


async def test_stats_counts_and_latency(
    client: httpx.AsyncClient, a_finished_call: str
) -> None:
    body = (await client.get("/api/stats")).json()
    assert body["total_calls"] == 1
    assert body["failed_calls"] == 1
    assert body["errors_24h"] == 1


async def test_stats_are_zero_not_null_on_an_empty_database(
    client: httpx.AsyncClient,
) -> None:
    """A fresh install must render, not crash the dashboard on nulls."""
    body = (await client.get("/api/stats")).json()
    assert body["total_calls"] == 0
    assert body["avg_tool_latency_ms"] == 0
    assert body["p95_tool_latency_ms"] == 0


async def test_limit_is_capped(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/calls?limit=10000")).status_code == 200


# --- current CRM details on the call list -----------------------------------
#
# A call row records the number that *rang*, frozen at the time. The contact's
# number can change afterwards, so a page showing only the frozen one disagrees
# with the CRM and reads as a bug. The list therefore enriches rows with the
# contact's current details — and must never let that take the page down.


async def test_call_list_shows_the_contacts_current_number(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.hubspot import Contact

    call_id = "api-crm-1"
    await calls.start_call(
        call_id=call_id, direction="inbound", from_number="+12025550100"
    )
    await calls.attach_contact(
        call_id=call_id, hubspot_contact_id="c-9", caller_name="Ada Okonkwo"
    )

    async def fake_batch(contact_ids: list[str]) -> dict[str, Contact]:
        return {
            "c-9": Contact(
                id="c-9",
                first_name="Ada",
                last_name="Okonkwo",
                email=None,
                # Moved on since the call.
                phone="+998998886810",
            )
        }

    monkeypatch.setattr("app.main._current_contacts", fake_batch)

    rows = (await client.get("/api/calls")).json()
    row = next(r for r in rows if r["call_id"] == call_id)
    assert row["from_number"] == "+12025550100", "what rang must not be rewritten"
    assert row["crm_phone"] == "+998998886810", "what the CRM holds now"


async def test_the_call_list_survives_the_crm_being_down(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The call history is local precisely so it outlives a CRM outage.

    Enriching it from HubSpot must not undo that: an outage is exactly when
    someone opens this page, and losing a current phone number is a fair price
    where losing the page is not.
    """
    call_id = "api-crm-2"
    await calls.start_call(call_id=call_id, direction="inbound", from_number="+1555")
    await calls.attach_contact(call_id=call_id, hubspot_contact_id="c-9")

    from app.services.hubspot import HubSpotClient

    async def exploding(self: HubSpotClient, contact_ids: list[str]) -> dict[str, str]:
        raise RuntimeError("hubspot is down")

    monkeypatch.setattr(HubSpotClient, "batch_read_contacts", exploding)
    monkeypatch.setattr("app.config.settings.hubspot_access_token", "pat-test")

    response = await client.get("/api/calls")
    assert response.status_code == 200
    row = next(r for r in response.json() if r["call_id"] == call_id)
    assert row["crm_phone"] is None
    assert row["from_number"] == "+1555", "the local record is unaffected"


async def test_no_crm_request_is_made_when_no_call_has_a_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unidentified callers are ordinary. Constructing a client to ask HubSpot
    about nobody would spend a round trip on every page load for nothing."""
    from app.main import _current_contacts
    from app.services.hubspot import HubSpotClient

    def exploding(*args: object, **kwargs: object) -> None:
        raise AssertionError("a CRM client was built with nothing to look up")

    monkeypatch.setattr(HubSpotClient, "__init__", exploding)

    assert await _current_contacts([]) == {}
    assert await _current_contacts(["", ""]) == {}
