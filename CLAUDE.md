# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with **uv**, not pip. `uv run` executes inside `.venv` without activating it. `uv.lock` is deliberately committed.

```bash
uv sync                                   # install deps from uv.lock
uv run python -m app.db                   # create/refresh schema only; safe to re-run
uv run python -m agent.main dev           # voice agent worker (LiveKit)
uv run uvicorn app.main:app --reload      # dashboard API on :8000
uv run python -m scripts.seed_erp         # catalogue, customers, orders (--reset wipes real ones)
uv run python -m scripts.seed_hubspot     # same customers into the CRM (--dry-run to preview)
uv run python -m scripts.seed_demo_call   # write a realistic finished call, no phone needed
uv run pytest                             # 214 tests; needs Postgres up
uv run pytest -k double_booking           # single test
uv run python -m evals.run                # score model behaviour (costs tokens; writes)
uv run python -m scripts.measure_latency  # where turn latency actually goes
uv run ruff check . && uv run mypy app agent   # both are clean; keep them that way
cd dashboard && npm run dev               # dashboard UI on :5173 (proxies /api to :8000)
```

**Postgres runs on 5434** — 5432 and 5433 are taken by other projects on this machine. `docker compose up -d` starts it. Tests use a separate `voice_crm_test` database, created automatically by `tests/conftest.py`.

**The test suite needs the container running.** `tests/conftest.py` sets `DATABASE_URL` *before* importing anything under `app.`, because `app.config` builds its `Settings` and `app.db` its engine at import time — that ordering is load-bearing, don't move the imports above it.

## Architecture

Three processes: the LiveKit agent worker (`agent/`), the dashboard API (`app/main.py`), and the React UI (`dashboard/`). They share the service layer and Postgres.

**The central rule: `app/services/` never imports LiveKit.** Services raise domain errors from `app/errors.py`; `agent/tooling.py` is the only place that translates them into LiveKit's `ToolError`. This is what makes the business logic unit-testable without a room, and it is why services return dataclasses rather than strings.

**`AGENT_MODE` picks between one agent and five; `unified` is the default.** It measured equal on evals and faster on latency (first turn 3.2s vs 5.0s), and it cannot have the two failure modes the split introduced: an unreachable capability, and a handoff loop. The unified agent inherits its tools from the specialists via the MRO, so the modes cannot drift; `tests/test_unified.py` fails if they do. Re-measure before switching — more tools may tip it back.

**In `specialists` mode, agents are specialists behind a router.** `ReceptionAgent` has no data tools at all — it only routes to `OrdersAgent`, `SalesAgent`, `SchedulingAgent` or `PolicyAgent`, each carrying two or three. That split exists because tool-selection accuracy degrades as the menu grows and degrades fastest on small models; one agent previously carried eight tools and confabulated. Handoff returns `(agent, message)` from a tool, copying `chat_ctx` with `exclude_instructions=True` so the caller never repeats themselves.

**Universal actions live on `BaseAgent`, not on a specialist.** `save_caller_details`, `log_call_summary`, `request_human_callback` and `return_to_reception` are on every agent. Putting a capability on one specialist makes it unreachable from the others: the model does not route to find a tool, it reaches for `request_human_callback` and promises a callback instead. Before adding a tool to a specialist, ask whether a caller might want it while talking to any of them.

**Handoffs are capped per caller turn.** Two agents will otherwise pass a request neither can serve back and forth forever, with the caller hearing nothing — that happened with "what furniture do you sell?", which no tool answered. `MAX_CONSECUTIVE_HANDOFFS` refuses the handoff and tells the model to act or escalate; the count resets whenever the caller speaks. The guard is a safety net, not a fix: a loop means a capability is missing.

**Anything that must survive a handoff lives in `CallState`**, carried as the session's `userdata` and reached through `context.userdata`. Held on an agent instance it would vanish mid-call — `offered_slots` is the clearest case, since the caller can detour to another specialist between hearing the times and choosing one.

**Handoffs are silent and self-answering.** Reception is told not to announce the transfer, and `BaseAgent.on_enter` generates a reply so the specialist picks up the pending question. Without the first, every route costs an extra LLM round trip and TTS pass; without the second, the caller has to repeat themselves.

**`agent/tooling.py` is the seam.** Every `@function_tool` body is wrapped by `@instrumented` (applied *under* `function_tool`, so `functools.wraps` lets `inspect.signature` still see the real signature — that signature and the docstring become the tool schema the model sees). It times the call, writes `tool_call`/`tool_result` rows to the timeline, and maps errors:

- `UserFacingError` → `ToolError(caller_message)`, logged INFO, **not** sent to Sentry. A caller asking about a nonexistent order is ordinary traffic; reporting it buries real failures.
- `IntegrationError` → `ToolError` with a vague message, logged ERROR, sent to Sentry. Upstream detail must never be read down a phone line.
- Anything else → generic apology. The catch-all is deliberately broad: an unhandled exception kills the turn and leaves the caller in silence.

**`call_id` is bound via context variables**, not passed as an argument, because LiveKit invokes tool callbacks for us and there is no parameter we control. `agent/tooling.current_call_id` and `app.observability.logging.bind_call_context` are set once per session in `agent/main.py`.

**HubSpot is the CRM; the ERP half is Postgres.** HubSpot models contacts and deals but has no stock levels or fulfilment state, so `app/services/erp.py` stands in behind the same interface. Swapping in Odoo/NetSuite means rewriting that module only.

**The dashboard shows both numbers, because they answer different questions.** `calls.from_number` is what *rang*, frozen at the time, and must never change — otherwise a caller who edits their number retroactively appears to have always called from the new one. The contact's number *now* lives in the CRM, so `/api/calls` reads it there: one batch request for the whole page, best-effort, degrading to `crm_phone: null` if HubSpot is unreachable. Showing only the frozen number makes the page look like it disagrees with HubSpot, which is what it looked like. Never let the CRM read take the page down — the call history is local precisely so it outlives an outage.

**Turn latency is the headline metric, tool latency is not.** Tool execution runs 4-7ms while turns run in seconds, so leading with tool latency reports the system as fast while the caller is sitting in silence. Turn timings arrive on `metrics_collected`, separately from transcripts, and are joined onto the `agent_speech` row in `_wire_transcripts` — that row is where a reader looks for "how long did this turn take". They were previously logged and never persisted, so the dashboard could not show them at all.

**Two records, two purposes.** HubSpot holds the business outcome (a note on the contact). `app/services/calls.py` holds the conversation — every utterance, tool call, latency and error — which is what the dashboard reads and what stays readable when HubSpot is down.

**Tests and evals answer different questions.** `tests/` covers deterministic logic — constraints, error mapping, retries, tool distribution — with a real Postgres, because most invariants worth protecting are enforced *by the database*. `evals/` covers model behaviour, which no test can pin: whether the model chooses to call a tool at all. Evals are scored, not passed, and cost tokens. Run them after touching a prompt, a model, or the tool split.

## Conventions

- **Config goes through `app.config.settings`.** Don't call `os.getenv()` anywhere else.
- **Raw SQL, no ORM models.** `app/db.py` holds the engine, `SCHEMA_STATEMENTS` (only ever creates) and `MIGRATION_STATEMENTS` (changes tables that already exist). The second list is the compromise the first one forced: a column added to a CREATE statement is invisible to every database that already has the table. `ADD COLUMN IF NOT EXISTS` keeps re-running safe but has no ordering, no history and no down path — affordable for a nullable column with no backfill and nothing else. A type change, a NOT NULL, a rename or any data migration is where it stops working and Alembic starts.
- **Anything indexing a migrated column goes in `MIGRATION_STATEMENTS` too**, not in `SCHEMA_STATEMENTS`. The creates run first, so an index over a column that only the migration adds fails outright on an existing database.
- **Multi-statement writes go through `db.transaction()`.** The other helpers open a session each, which is right for one statement and wrong for several that depend on each other — `place_order` writes an order, its lines and a stock decrement, and any one landing alone is worse than the call failing.
- **`scheduling.book()` relies on the UNIQUE constraint on `appointments.starts_at`** to prevent double-booking under concurrency. The free/busy read is advisory only. Preserve that shape — don't replace it with a read-then-write.
- **`calls.record_event` swallows its own failures by design.** Losing a timeline row is regrettable; dropping a live call because a timeline write failed is not.
- **Never block the event loop.** `httpx` async for outbound HTTP, `asyncpg` via SQLAlchemy async for Postgres.
- **Tool docstrings are prompts.** The summary line and `Args:` entries become the schema the model sees — write them for the model, not for a reader of the source.
- **Agent instructions are deliberately unemphatic.** Current models follow the system prompt closely, so `CRITICAL: you MUST call the tool` over-triggers. Say when a tool applies and stop.
- **Everything the agent says is read aloud.** No markdown, no bullets, no emoji, no SKUs unless the caller used one. The speech-shaping block in each agent's instructions is load-bearing.
- Background tasks spawned from LiveKit event handlers must be held in a set until done — asyncio keeps only a weak reference, so a bare `create_task` can be garbage-collected mid-write (see `_wire_transcripts`).
- Commit messages carry **no Claude attribution** — no `Co-Authored-By`, no "Generated with" trailer.

## Gotchas

- **A client is a HubSpot contact; there is no customer table.** Postgres holds orders keyed by phone number, not customers, so "adding a client" means creating a contact and nothing else. `save_caller_details` is the only path that does it during a call — it updates when `state.contact` is set and creates when it is not. A contact created with no usable number can never be found again, because both identification and order lookup match on it.

- **The phone number is the join key between the CRM and the ERP.** Identification looks the caller up in HubSpot by number; order lookup matches the same number in Postgres. Change it on one side only and the caller is greeted by name and told they have no orders, which reads as a broken lookup. `save_caller_details` moves the orders with `erp.relink_orders` when the number changes.
- **Deals are not orders.** A deal is a CRM opportunity in HubSpot; an order is a placed order in the ERP. They count different things, so a contact with one deal and two orders is not a bug. `record_purchase_interest` opens the deal, `place_order` closes it won *and* writes the order, linking the two through `orders.hubspot_deal_id`. That link is the only thing joining the halves besides the phone number, and it is nullable — most orders never came from a call.

- **`place_order` takes stock; `record_purchase_interest` does not.** The two sit next to each other on the sales agent and only the model chooses between them, from nothing but how the caller phrased it. `evals/cases.py` covers both directions, and the hesitation case asserts *restraint* — that `place_order` did not fire — rather than demanding a particular call, because asking a hesitant caller a clarifying question is a good answer too.

- **Overselling is prevented by the database, like double-booking.** The decrement carries its own `stock_qty >= quantity` guard, so two callers racing for the last item means one UPDATE matches no row. The `CHECK (stock_qty >= 0)` constraint is the backstop, not the mechanism. Don't replace it with a read-then-write.

- **A CRM failure must not lose a placed order.** The ERP is the system of record for the sale; `place_order` writes it first and closes the deal best-effort, logging rather than raising. A caller who has been read an order reference must not have that order vanish because HubSpot was down.
- **`fetch_all` does not commit.** It is for reads. An `UPDATE ... RETURNING` run through it reports the rows it changed and then discards them when the session closes — success everywhere except the database. Use `execute_returning`.

- **Dictated numbers are validated by libphonenumber, not by length.** `normalise_spoken_phone` refuses anything `phonenumbers` calls invalid. A range check is not enough and the gap is not theoretical: `+999899999` is nine digits, passes any length rule, and was accepted off a real call — but `+999` is not an assigned country code, so the caller's orders were relinked to a number nobody can ring from. Note the deliberate asymmetry with `phone_variants`: **reading is lenient, writing is strict.** A number already in the CRM was typed by a human and must keep matching; a new number is the moment to refuse, because that is when a mishearing becomes permanent.
- **Demo numbers live in `+1 202 555 01xx`**, the fictional block that is still well formed. They were `+1555010x`, which libphonenumber rejects — so a demo caller could be *identified* but could not dictate their own number back, and the number-change flow could not be demonstrated against the demo data. `tests/test_demo_data.py` pins it.

- **Evals write.** `sales.commit` places a real order and consumes real stock, so `evals/run.py` points `DATABASE_URL` at a throwaway database via `evals/scratch_db.py`, imported *before* anything under `app.` for the same load-bearing reason as `tests/conftest.py`. There is no equivalent escape hatch for HubSpot: evals create deals and notes in whatever portal the token names.

- **Anything a caller dictates arrives as words, not as data.** "ada at example dot com" is what speech recognition produces, and HubSpot rejects it with `INVALID_EMAIL`, failing the whole create — name and number lost with it. `normalise_spoken_email` repairs it and returns `None` when it cannot, so the field is dropped rather than the record. Expect the same class of problem for any dictated identifier: order references, postcodes, card numbers.

- **Console mode: don't name a hardware audio device.** `--input-device "RODE"` picks the raw ALSA device, which cannot resample; many USB mics are 48 kHz only while the agent opens at 24 kHz, giving `Invalid sample rate [PaErrorCode -9997]`. Omit the flag, or use `pulse`/`default` so PipeWire resamples. On Linux, `libportaudio2` must also be installed — the CLI validates audio devices before it checks `--text`, so a missing library blocks text mode too.

- **`AgentServer` registers exactly one `rtc_session`.** Both directions share one entrypoint named `voice-agent`, which branches on job metadata: an outbound dispatch carries a `phone` to dial, an inbound SIP job does not. The inbound dispatch rule and `scripts/run_campaign.py` must both target that same agent name.

- **Associate records with the v4 default endpoint**, `PUT /crm/v4/objects/{type}/{id}/associations/default/{type}/{id}` — no body, no `associationTypeId`. Those ids differ per object pair, and a wrong one fails *silently*, leaving an orphaned record that never appears on the contact. `log_call_note` is the one place still using an inline type id (202), which works but is the weaker pattern.
- **HubSpot deals have no natural key**, so seeding checks by name before creating. Contacts dedupe by phone; deals will happily stack duplicates on every re-run without the check.

- **`scripts/demo_data.py` is the single source for the demo dataset**, read by both `seed_erp` and `seed_hubspot`. `app/db.py` seeded a rival catalogue until it was removed: three products existed under two SKUs each and ORD-5001 was written twice with different phone numbers, so whichever seeder ran first won. Splitting stock across two rows for one real product makes neither row true — `check_stock` can report on one while `place_order` decrements the other. `tests/test_demo_data.py` pins uniqueness; `scripts/dedupe_products.py` repairs a database that already has the split.
- **`seed_erp` is additive by default; `--reset` deletes.** Orders are no longer only ever seeded — `place_order` writes real ones during a call, and re-seeding also resets `contact_phone` on every order, silently undoing a number change made by `save_caller_details`. The phone numbers must agree across the two: identification reads HubSpot, order lookup reads Postgres. Seeded separately, you get a caller greeted by name whose order cannot be found, which looks like an agent bug.
- **HubSpot needs a Private App token (`pat-…`).** An OAuth access token (`CiRl…`, ~30 min lifetime) fails with `expired 20677 day(s) ago` and `"expire time": 1970-01-01` — epoch zero, meaning "invalid", not "expired".
- Bound parameters arrive untyped over asyncpg, so `CURRENT_DATE + :days` fails with `operator is not unique: date + unknown`. Cast explicitly: `CURRENT_DATE + CAST(:days AS integer)`.

- The Postgres image is `pgvector/pgvector:pg16`, not `postgres:16`, because the knowledge base stores embeddings alongside everything else. Switching image can trigger `collation version mismatch` warnings on an existing volume — the images ship different glibc versions. Fix with `ALTER DATABASE <db> REFRESH COLLATION VERSION` for `voice_crm`, `voice_crm_test`, `voice_crm_evals`, `postgres` **and `template1`** — `CREATE DATABASE` copies from `template1`, so a mismatch there fails every new database with an `InternalServerError` that names the template rather than the database being created; ignoring it risks corrupt text indexes.
- `knowledge.MAX_DISTANCE` is calibrated against the seeded corpus, not derived. Relevant questions score 0.26–0.39 and unrelated ones ~0.60; the cutoff sits between. Materially change the corpus and it needs re-checking — `tests/test_knowledge.py` pins both ends of that band so it fails loudly rather than the agent answering nonsense.

- LiveKit's API changed in 1.x: it is `AgentServer` + `@server.rtc_session` + `inference.STT/LLM/TTS`, **not** the older `WorkerOptions`/plugin-per-vendor shape. Check the installed package before writing against remembered APIs.
- `mypy` has no `python_version` pin on purpose — pinning it below the venv's interpreter makes third-party stubs using newer syntax fail to parse, which silently halts all checking.
- The dashboard's tsconfig sets `erasableSyntaxOnly`, so TypeScript constructor parameter properties (`constructor(readonly x: number)`) don't compile. Declare and assign explicitly.
- **LiveKit Inference does not host Anthropic or Meta models.** Valid prefixes are `google/`, `openai/`, `xai/`, `deepseek/`, `kimi/` — an `anthropic/...` string fails at runtime. Using Claude means bypassing Inference with `livekit-plugins-anthropic` and your own key. Check https://docs.livekit.io/agents/models/llm/ before changing `LLM_MODEL`.
- **Latency is network, not code.** ~1.5s per model round trip to LiveKit Inference; tool execution is 4–7ms. Routing costs one extra round trip on the *first* turn only (≈5s vs ≈2.2s thereafter). Measured: model swaps make it worse, prompt size does nothing. `preemptive_generation` and `endpointing.min_delay` in `_build_session` are the levers that work.
- `LLM_MODEL` defaults to `google/gemma-4-31b-it` — LiveKit's latency-optimised default, and the cheapest. It is weaker at tool calling than a frontier model, which is what this agent depends on; judge it against a real conversation, not in the abstract.
