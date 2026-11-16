"""Single source of truth for configuration.

Nothing else in the codebase calls ``os.getenv`` — import ``settings`` instead.
That keeps the full set of required environment variables discoverable in one
place and lets tests override values without touching the process environment.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LiveKit -----------------------------------------------------------
    livekit_url: str = Field(default="", alias="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", alias="LIVEKIT_API_SECRET")
    # Required only for outbound calls. Create with:
    #   lk sip outbound create --name <name> --address <provider> --numbers <did>
    sip_outbound_trunk_id: str = Field(default="", alias="SIP_OUTBOUND_TRUNK_ID")

    # --- HubSpot -----------------------------------------------------------
    # Private app access token (pat-...). Scopes needed: crm.objects.contacts
    # read/write, crm.objects.deals read/write, and crm.objects.owners read.
    hubspot_access_token: str = Field(default="", alias="HUBSPOT_ACCESS_TOKEN")
    hubspot_base_url: str = Field(
        default="https://api.hubapi.com", alias="HUBSPOT_BASE_URL"
    )

    # --- Database ----------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5434/voice_crm",
        alias="DATABASE_URL",
    )

    # --- Model -------------------------------------------------------------
    # All three run through LiveKit Inference, so they need no provider API
    # keys of their own — LiveKit credentials cover them.
    #
    # Gemma 4 31B is open-weight, served on LiveKit's own infrastructure, and
    # is their recommended default for voice because it is latency-optimised.
    #
    # The tradeoff is tool calling: this agent is only useful if the model
    # reliably picks the right tool and fills its arguments correctly, and a
    # small open model is weaker at that than a frontier model. If you see it
    # inventing order references, skipping tools, or passing a slot_id that was
    # never offered, move to a stronger model.
    #
    # Valid prefixes on LiveKit Inference are the provider families it hosts:
    # google/, openai/, xai/, deepseek/, kimi/. There is NO anthropic/ or
    # meta/ — Claude and Llama are not offered through Inference. Current list:
    # https://docs.livekit.io/agents/models/llm/
    #
    # To use Claude anyway, bypass Inference with the Anthropic plugin
    # (livekit-plugins-anthropic) in agent/main.py `_build_session()`, which
    # needs your own ANTHROPIC_API_KEY.
    llm_model: str = Field(default="google/gemma-4-31b-it", alias="LLM_MODEL")
    stt_model: str = Field(default="deepgram/nova-3", alias="STT_MODEL")
    # "multi" runs a multilingual model, which is slower than a single-language
    # one. Set to "en" (or your language) unless callers actually switch.
    stt_language: str = Field(default="en", alias="STT_LANGUAGE")
    tts_model: str = Field(default="inworld/inworld-tts-2", alias="TTS_MODEL")
    tts_voice: str = Field(default="Ashley", alias="TTS_VOICE")

    # Seconds of silence before the agent decides the caller has finished.
    # Pure dead air on every turn; lower feels sharper, too low interrupts
    # people who pause to think.
    endpointing_min_delay: float = Field(default=0.3, alias="ENDPOINTING_MIN_DELAY")

    # Console mode has no caller ID, so the agent always meets a stranger and
    # the whole identification path goes untested. Set this to a seeded number
    # to exercise it locally. Ignored on real SIP calls, which carry their own.
    console_caller_phone: str = Field(default="", alias="CONSOLE_CALLER_PHONE")

    # "unified": one agent holding every tool. "specialists": reception routes
    # to four narrow agents.
    #
    # Unified is the default because it measured better on both axes that
    # matter. Same eval score, first turn 3.2s against 5.0s, and none of the
    # failure modes the split introduced — a capability reachable from only
    # one agent, or two agents passing a request neither could serve. The
    # split was adopted on the principle that shorter tool menus are chosen
    # from more accurately; that did not show up in measurement here, on 11
    # tools with a small model.
    #
    # Re-measure before changing it: `AGENT_MODE=specialists uv run python -m
    # evals.run` and `scripts/measure_latency.py`. More tools may tip it back.
    agent_mode: str = Field(default="unified", alias="AGENT_MODE")

    # --- Observability -----------------------------------------------------
    sentry_dsn: str = Field(default="", alias="SENTRY_DSN")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # Human-readable console logs locally; JSON everywhere else.
    log_json: bool = Field(default=False, alias="LOG_JSON")
    environment: str = Field(default="development", alias="ENVIRONMENT")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
