"""State shared across every agent on a call.

Specialists hand off to each other mid-conversation, so anything that must
survive a handoff lives here rather than on an agent instance. LiveKit carries
it as the session's ``userdata``, reachable from any tool through
``context.userdata``.

The offered appointment slots are the clearest example: scheduling reads them
out, the caller says "the first one", and the booking may well happen after a
detour through another agent. Held on the agent, that mapping would vanish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.services.hubspot import Contact, HubSpotClient


@dataclass
class CallState:
    call_id: str
    caller_phone: str | None = None
    #: None when no CRM is configured — every CRM path checks this.
    hubspot: HubSpotClient | None = None
    contact: Contact | None = None
    #: Slots read out to the caller, by the short id used to refer to them.
    #: Booking takes that id rather than a timestamp, so the model never has to
    #: do date arithmetic — a reliable source of invented times.
    offered_slots: dict[int, datetime] = field(default_factory=dict)
    #: Set once the summary has been written, so the shutdown handler can tell
    #: a completed call from one the caller abandoned.
    summary_logged: bool = False
    #: Outbound only: the deal this call is following up, so qualification can
    #: move it to the right stage.
    deal_id: str | None = None
    #: Handoffs since the caller last said anything. Two agents can otherwise
    #: bounce a request neither can serve back and forth forever, with the
    #: caller listening to silence — reset on every user turn.
    handoffs_since_user_turn: int = 0

    @property
    def crm_connected(self) -> bool:
        return self.hubspot is not None
