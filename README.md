# AI Voice Agents for CRM & ERP

Phone agents that do real work in business systems: identify the caller, answer
order and stock questions, book appointments, qualify outbound leads, and write
everything back to the CRM — with a dashboard showing exactly what the agent
said, which tools it called, how long each took, and what failed.

- **Voice** — [LiveKit Agents](https://docs.livekit.io/agents/) (`AgentServer` +
  `AgentSession`), so the realtime audio loop, turn detection and barge-in are
  handled and the agent logic stays ours.
- **CRM** — HubSpot. The agent reads *and writes*: identifies the caller,
  creates and corrects contacts, opens deals when someone wants to buy, raises
  follow-up tasks when a human is needed, and logs a call summary.
- **ERP** — inventory and orders in Postgres, behind the same service interface
  as the CRM so it can be swapped for Odoo/NetSuite without touching the tools.
- **Knowledge base** — policy questions (returns, warranty, delivery) answered by
  retrieval over pgvector, with embeddings computed locally by fastembed so the
  lookup costs milliseconds rather than a network round trip.
- **Dashboard** — React + Vite, reading a FastAPI API over the call history.
- **One agent, or five** — `AGENT_MODE` switches between a single agent
  holding every tool (the default) and a reception agent routing to four
  specialists. See *Agent modes* below for why the default is what it is.

## Quickstart

```bash
cp .env.example .env          # fill in LiveKit credentials (HubSpot is optional)
docker compose up -d          # Postgres on 5434
uv sync
uv run python -m app.db       # create the schema
uv run python -m scripts.seed_erp         # 14 products, 5 customers, 5 orders
uv run python -m scripts.seed_knowledge   # index the policy documents (RAG)
uv run python -m scripts.seed_hubspot     # same customers into HubSpot (optional)
```

`scripts/demo_data.py` is the single source for the demo dataset, and both
seeders read it. That is deliberate: caller identification looks a number up in
HubSpot while order lookup matches the same number in Postgres, so seeding them
from separate lists produces a caller the agent greets by name but whose order
it cannot find — which reads as an agent bug rather than a fixture mismatch.

`seed_hubspot` matches on phone before creating, so it is safe to re-run
against a real portal. `--dry-run` prints what it would write.

**Getting the HubSpot token:** it is under **Development**, not Settings >
Integrations — either `Development > Keys > Service Keys` (simplest, meant for
API-only access) or `Development > Legacy Apps` (the older private app flow,
still supported). An OAuth access token will not work: those look like `CiRl…`,
last about 30 minutes, and fail with a misleading "expired 20677 days ago".

On Linux, console mode also needs PortAudio — `sounddevice` dlopens it and the
CLI validates audio devices before it checks the `--text` flag, so a missing
library blocks text mode too:

```bash
sudo apt install -y libportaudio2      # Debian/Ubuntu
```

Let it pick the default audio device. Naming a USB microphone directly
(`--input-device "RODE"`) selects the raw ALSA device, which does not resample
— and many USB mics are 48 kHz only while the agent opens the stream at 24 kHz,
so PortAudio fails with `Invalid sample rate [PaErrorCode -9997]`. The default
device goes through PipeWire or PulseAudio, which converts transparently.

## Connecting a phone number

Buy a number in the LiveKit Cloud dashboard (Telephony → Phone Numbers), or
bring one from a provider. Everything after that is scripted:

```bash
uv run python -m scripts.setup_telephony --show              # what exists now
uv run python -m scripts.setup_telephony --inbound +1415...  # trunk + dispatch rule
uv run python -m agent.main start                            # then call it
```

The dispatch rule's `agent_name` must match the name the worker registers
under. `setup_telephony` imports `AGENT_NAME` from the worker rather than
repeating the string, because a mismatch fails in the least helpful way
possible: the call connects, a room is created, and nobody joins — the caller
hears silence with no error anywhere.

## Try it without a phone number

`console` mode runs the whole agent against your terminal — no SIP trunk, no
number, no LiveKit room. Only `LIVEKIT_URL`, `LIVEKIT_API_KEY` and
`LIVEKIT_API_SECRET` are required.

```bash
uv run python -m agent.main console --text   # type to it; isolates tool calling
uv run python -m agent.main console          # talk to it; this is what measures latency
```

Start with `--text`. It answers the question that decides your model — does it
call the right tool with the right arguments — without any audio setup.

Three processes, three terminals:

```bash
uv run python -m agent.main dev                  # the voice agent worker
uv run python -m scripts.run_campaign --dry-run  # outbound: who would be called
uv run uvicorn app.main:app --reload             # dashboard API on :8000
cd dashboard && npm install && npm run dev       # dashboard UI on :5173
```

No phone line handy? Seed a realistic finished call so the dashboard has
something to show:

```bash
uv run python -m scripts.seed_demo_call
```

## Tests

```bash
docker compose up -d                # the suite needs Postgres
uv run pytest                       # 199 tests
uv run pytest tests/test_erp.py     # one file
uv run pytest -k double_booking     # one test
uv run ruff check . && uv run mypy app agent
```

### Evals — for the part tests cannot cover

```bash
uv run python -m evals.run                              # score current behaviour
uv run python -m evals.run --model openai/gpt-4.1-mini  # compare models
uv run python -m evals.run --repeat 3                   # sample the variance
```

A test can assert that `check_stock` returns the right row. It cannot assert
that the *model decides to call it* when someone says "do you sell office
chairs?" — that is not deterministic, so it is scored rather than passed. Every
case in `evals/cases.py` corresponds to a real failure seen in development.
Run it after changing a prompt, a model, or the tool split: the three things
that change behaviour without breaking a test.

Postgres is real in tests, not mocked — most of the invariants worth protecting
are enforced *by the database* (the UNIQUE constraint that prevents double
booking, the CHECK constraints on status columns), so mocking it would mock away
the thing under test. HubSpot is mocked at the transport layer with `respx`, so
the retry and error-mapping code still runs.

## How it fits together

```
 phone ──SIP──▶ LiveKit ──▶ agent/main.py ──▶ UnifiedAgent
                                                  │  (or ReceptionAgent +
                                                  │   4 specialists, per
                                                  │   AGENT_MODE)
                                                  │ @function_tool
                                                  ▼
                                          agent/tooling.py
                                    (timing · timeline · ToolError)
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          ▼                       ▼                       ▼
                 services/hubspot.py      services/erp.py      services/scheduling.py
                     (HubSpot)              (Postgres)              (Postgres)
                          └───────────────────────┼───────────────────────┘
                                                  ▼
                                          services/calls.py
                                     (call + timeline history)
                                                  │
                                    app/main.py ──▶ dashboard/
```

**Services never import LiveKit.** They raise domain errors from `app/errors.py`;
`agent/tooling.py` is the only place that translates those into LiveKit's
`ToolError`. That keeps every piece of business logic unit-testable without
spinning up a room, and it is what makes the error behaviour predictable:

| Error | Caller hears | Logged as | Sentry |
|---|---|---|---|
| `UserFacingError` | the specific reason ("no order with that reference") | INFO | no |
| `IntegrationError` | a vague apology and a fallback offer | ERROR | yes |
| anything unhandled | a generic apology, turn continues | ERROR + traceback | yes |

User errors are excluded from Sentry deliberately: a caller asking about an
order that doesn't exist is ordinary traffic, and reporting it would bury real
integration failures.

## Logging

Structured JSON via `structlog`. Every line emitted during a call carries the
same `call_id`, bound once per session with context variables — LiveKit invokes
tool callbacks for us, so there is no argument we control to thread it through.

```
[info ] tool_call    call_id=RM_x7 tool=check_stock product='desk lamp'
[info ] tool_result  call_id=RM_x7 tool=check_stock elapsed_ms=41
```

Set `LOG_JSON=true` outside local development.

## Agent modes

The obvious design for a multi-skill voice agent is to split it into
specialists behind a router, on the principle that shorter tool menus are
chosen from more accurately. That was built first, and measured second:

| | specialists | unified |
|---|---|---|
| eval score | 10/10 | 10/10 |
| first turn | 4993 ms | **3217 ms** |
| turns after | 2180 ms | 2853 ms |
| first-turn penalty | 2.3x | 1.1x |

Same accuracy, and the routing round trip is pure cost on the first turn. The
split also introduced two failure modes that one agent cannot have: a
capability reachable from only one specialist (a caller asking to correct their
name was passed around instead of helped), and two agents handing a request
neither could serve back and forth until the caller gave up.

So `unified` is the default. The split is still there, still tested, and worth
re-measuring if the tool count grows well past a dozen:

```bash
AGENT_MODE=specialists uv run python -m evals.run
AGENT_MODE=specialists uv run python -m scripts.measure_latency
```

Both modes share their tool definitions — the unified agent inherits them from
the specialists — so a tool added to one appears in the other, and a test
fails if they diverge.

## Latency

Voice is unforgiving about delay, and on this deployment the dominant cost is
network distance to LiveKit's inference endpoint — roughly 1.5s per model round
trip, from Tashkent. Tool execution is 4–7ms, so the code is not the problem.

```bash
uv run python -m scripts.measure_latency
```

```
turn 1   4993 ms   to_sales, check_stock     first turn — routes to sales
turn 2   2309 ms   check_stock               same specialist
turn 3   2051 ms   check_stock               same specialist
```

**Routing is paid once per call, not per turn.** The first turn spends an extra
round trip deciding where to send the caller; every turn after it stays on the
specialist and costs about half as much.

Things that were measured and did *not* help: switching models (Gemma is the
fastest tested — Gemini Flash Lite and GPT-4.1-mini are both slower), and
shortening prompts (a three-word prompt takes the same time as a 1,200-token
one). Things that do: preemptive generation and TTS, which overlap a round trip
with speech that is happening anyway; a lower endpointing delay; and, if you
have a GPU, a local model, which removes the network hop entirely.

## Notes and limitations

- **HubSpot has no ERP half.** It models contacts and deals but not stock levels
  or fulfilment, so inventory and orders live in Postgres behind
  `app/services/erp.py`. Swapping in a real ERP means rewriting that module and
  nothing above it.
- **Model choice trades latency against tool-calling reliability.** `LLM_MODEL`
  defaults to `google/gemma-4-31b-it` — LiveKit's latency-optimised default and
  the cheapest option. Voice is latency-sensitive and every extra second is dead
  air on the line, but this agent is only useful if the model reliably picks the
  right tool and fills its arguments; small models are weaker at that. Note
  LiveKit Inference hosts `google/`, `openai/`, `xai/`, `deepseek/` and `kimi/`
  models — **not Anthropic or Meta Llama**. Using Claude means bypassing
  Inference with `livekit-plugins-anthropic` and your own API key.
- **Outbound calls need an outbound SIP trunk.** `scripts/run_campaign.py`
  finds open deals in HubSpot and dispatches a call per lead; the worker dials
  via `ctx.add_sip_participant`. Set `SIP_OUTBOUND_TRUNK_ID` first. Calls are
  placed serially with a delay — dialling a list in parallel is how a bug
  becomes a hundred unwanted calls before anyone notices.
- **Phone matching is best-effort.** Caller ID arrives in E.164; HubSpot records
  are hand-entered. `phone_variants()` tries several normalisations, and an
  unmatched number is treated as a new caller rather than an error.
- **Appointments are local.** Slots live in Postgres, not Google Calendar or
  HubSpot Meetings.
