"""Dashboard API.

Read-only over the call history the agent writes. The call record is local and
stays readable when HubSpot is down — which is exactly when you most want to
look at it.

The one CRM read is deliberate and narrow. A call row records the number that
*rang*, frozen at the time; the contact's number can change afterwards, and a
page showing only the frozen one disagrees with the CRM and reads as a bug. So
the list enriches rows with the contact's current details, in a single batch
request for the whole page, best-effort: if HubSpot is unreachable the rows
lose their current number and nothing else. Never let the CRM take the page
down with it.

Run with ``uv run uvicorn app.main:app --reload``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.db import engine
from app.errors import AgentError, NotFoundError
from app.observability.logging import configure_logging, get_logger
from app.observability.sentry import capture, init_error_reporting
from app.services import calls
from app.services.hubspot import HubSpotClient

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    init_error_reporting()
    log.info("api_starting")
    yield
    await engine.dispose()
    log.info("api_stopped")


app = FastAPI(title="Voice Agent Dashboard", version="0.1.0", lifespan=lifespan)

# The Vite dev server runs on a different origin during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(NotFoundError)
async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": exc.message})


@app.exception_handler(AgentError)
async def _agent_error(request: Request, exc: AgentError) -> JSONResponse:
    log.error("api_error", path=request.url.path, reason=exc.message, **exc.context)
    capture(exc, path=request.url.path)
    return JSONResponse(status_code=502, content={"detail": exc.message})


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stats")
async def get_stats() -> dict[str, Any]:
    return await calls.stats()


async def _current_contacts(contact_ids: list[str]) -> dict[str, Any]:
    """Current CRM details for the rows on this page, or nothing.

    Best-effort by design: the call history is local precisely so it survives
    the CRM being down, and enriching it must not undo that.
    """
    unique = sorted({cid for cid in contact_ids if cid})
    if not unique or not settings.hubspot_access_token:
        return {}
    client = HubSpotClient()
    try:
        return await client.batch_read_contacts(unique)
    except Exception as exc:  # noqa: BLE001 — see docstring
        log.warning("crm_enrichment_failed", error=str(exc), contacts=len(unique))
        return {}
    finally:
        await client.aclose()


@app.get("/api/calls")
async def list_calls(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    rows = await calls.list_calls(limit=min(limit, 200), offset=offset)
    # Both resolved once for the whole page rather than per row: the list is
    # where someone scans for "did anything come of these", and a request per
    # call would make it the slowest page in the app.
    produced = await calls.outcomes([row.call_id for row in rows])
    contacts = await _current_contacts([row.hubspot_contact_id or "" for row in rows])
    enriched: list[dict[str, Any]] = []
    for row in rows:
        contact = contacts.get(row.hubspot_contact_id or "")
        enriched.append(
            {
                **vars(row),
                "outcomes": produced.get(row.call_id, []),
                # What the CRM says *now*. None when the contact is unknown,
                # deleted, or HubSpot could not be reached.
                "crm_phone": contact.phone if contact else None,
                "crm_name": contact.display_name if contact else None,
            }
        )
    return enriched


@app.get("/api/calls/{call_id}")
async def get_call(call_id: str) -> dict[str, Any]:
    return vars(await calls.get_call(call_id))


@app.get("/api/calls/{call_id}/events")
async def get_call_events(call_id: str) -> list[dict[str, Any]]:
    # 404 rather than an empty list for an unknown call, so the UI can tell
    # "no such call" apart from "call with nothing recorded yet".
    await calls.get_call(call_id)
    return await calls.get_events(call_id)


@app.get("/api/errors")
async def list_errors(limit: int = 50) -> list[dict[str, Any]]:
    return await calls.recent_errors(limit=min(limit, 200))
