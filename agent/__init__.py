"""Agent package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.base import BaseAgent
    from agent.state import CallState


def entry_agent(state: CallState) -> BaseAgent:
    """The agent that takes an inbound call, per ``AGENT_MODE``.

    One place so the worker and the evals cannot disagree about which
    configuration is being run.
    """
    from app.config import settings

    if settings.agent_mode == "unified":
        from agent.unified import UnifiedAgent

        return UnifiedAgent(state=state)

    from agent.specialists import ReceptionAgent

    return ReceptionAgent(state=state)


__all__ = ["entry_agent"]
