import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "./api";
import type { Call, CallEvent, ErrorRow, Outcome, Stats } from "./types";

const POLL_MS = 5000;

/** Latency bar width. Capped so one 5s outlier doesn't flatten every other row. */
function barWidth(ms: number): number {
  return Math.max(2, Math.min(120, Math.round((ms / 1500) * 120)));
}

function clockTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function relativeTime(iso: string): string {
  const seconds = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return new Date(iso).toLocaleDateString([], { month: "short", day: "numeric" });
}

function duration(call: Call): string {
  if (!call.ended_at) return "live";
  const ms = new Date(call.ended_at).getTime() - new Date(call.started_at).getTime();
  const s = Math.round(ms / 1000);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

function phoneNumber(call: Call): string {
  return call.direction === "inbound"
    ? (call.from_number ?? "unknown caller")
    : (call.to_number ?? "unknown number");
}

/** Who the row is about. The CRM's current name wins, then the name recorded
 *  at the time, then the bare number. */
function who(call: Call): string {
  return call.crm_name ?? call.caller_name ?? phoneNumber(call);
}

/** True when the contact's number has changed since this call came in.
 *
 *  Both numbers are correct and they are answers to different questions:
 *  "what rang" is a fact about the call and must never change, "what is their
 *  number" is a fact about the contact and does. Showing only the first makes
 *  the page look like it disagrees with the CRM. */
function numberChangedSince(call: Call): boolean {
  return Boolean(call.crm_phone && call.crm_phone !== phoneNumber(call));
}

const OUTCOME_LABELS: Record<string, string> = {
  order_placed: "order",
  purchase_interest: "opportunity",
  appointment_booked: "booking",
};

/** One chip per business result, so a call that sold something looks different
 *  from one that answered a question. Both end "completed". */
function OutcomeChips({ outcomes }: { outcomes: Outcome[] }) {
  if (!outcomes || outcomes.length === 0) return null;
  return (
    <span style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
      {outcomes.map((outcome, index) => (
        <span className="tag outcome" key={`${outcome.kind}-${index}`}>
          {OUTCOME_LABELS[outcome.kind] ?? outcome.kind}
          {outcome.order_ref ? ` ${outcome.order_ref}` : ""}
        </span>
      ))}
    </span>
  );
}

/** Tool args, minus the noise, rendered for a single line. */
function formatArgs(payload: Record<string, unknown>): string {
  const entries = Object.entries(payload).filter(
    ([key]) => !["outcome", "result", "reason", "type"].includes(key),
  );
  if (entries.length === 0) return "";
  return entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ");
}

export default function App() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [calls, setCalls] = useState<Call[]>([]);
  const [errors, setErrors] = useState<ErrorRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [events, setEvents] = useState<CallEvent[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, c, e] = await Promise.all([api.stats(), api.calls(), api.errors()]);
      setStats(s);
      setCalls(c);
      setErrors(e);
      setLoadError(null);
      setSelected((current) => current ?? c[0]?.call_id ?? null);
    } catch (err) {
      setLoadError(err instanceof ApiError ? err.message : "Something went wrong.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    void api
      .events(selected)
      .then((rows) => !cancelled && setEvents(rows))
      .catch(() => !cancelled && setEvents([]));
    return () => {
      cancelled = true;
    };
  }, [selected, calls]);

  const current = calls.find((c) => c.call_id === selected) ?? null;

  return (
    <div className="shell">
      <header className="masthead">
        <h1>Call Ledger</h1>
        <span className="sub">Voice agent operations · CRM &amp; ERP</span>
      </header>

      {loadError ? (
        <p className="state error">
          {loadError} Start it with <code>uv run uvicorn app.main:app --reload</code>.
        </p>
      ) : null}

      <Figures stats={stats} />

      <div className="panes">
        <section>
          <h2 className="pane-title">Calls</h2>
          {calls.length === 0 && !loadError ? (
            <p className="state">
              No calls yet. Seed one with <code>uv run python -m scripts.seed_demo_call</code>.
            </p>
          ) : (
            <ul className="call-list">
              {calls.map((call) => (
                <li key={call.call_id}>
                  <button
                    type="button"
                    className="call-row"
                    aria-current={call.call_id === selected}
                    onClick={() => setSelected(call.call_id)}
                  >
                    <span className="line-1">
                      <span className="who">{who(call)}</span>
                      <span className="when">{relativeTime(call.started_at)}</span>
                    </span>
                    <span className="line-1" style={{ marginTop: 2 }}>
                      <span className="when">
                        {phoneNumber(call)}
                        {numberChangedSince(call) ? ` · now ${call.crm_phone}` : ""}
                      </span>
                    </span>
                    <span className="line-1" style={{ marginTop: 6 }}>
                      <span style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        <span className={`tag ${call.status}`}>
                          {call.status.replace("_", " ")}
                        </span>
                        <span className="tag dir">{call.direction}</span>
                        <OutcomeChips outcomes={call.outcomes} />
                      </span>
                      <span className="when">{duration(call)}</span>
                    </span>
                    {call.summary ? <span className="line-2">{call.summary}</span> : null}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section>
          <h2 className="pane-title">Transcript &amp; actions</h2>
          <CallDetail call={current} events={events} />
        </section>
      </div>

      <section className="errors">
        <h2 className="pane-title">Recent errors</h2>
        {errors.length === 0 ? (
          <p className="state">No errors recorded. </p>
        ) : (
          errors.map((row) => (
            <div className="error-row" key={row.id}>
              <span className="type">{String(row.payload.type ?? "Error")}</span>
              <span className="reason">
                {String(row.payload.reason ?? "")} {row.name ? `· ${row.name}` : ""}
              </span>
              <button
                type="button"
                className="call-link"
                onClick={() => setSelected(row.call_id)}
              >
                {row.call_id}
              </button>
              <span style={{ color: "var(--ink-faint)" }}>{clockTime(row.created_at)}</span>
            </div>
          ))
        )}
      </section>
    </div>
  );
}

function Figures({ stats }: { stats: Stats | null }) {
  const items: { label: string; value: string; tone?: string }[] = stats
    ? [
        { label: "Calls · 24h", value: String(stats.calls_24h) },
        { label: "Live now", value: String(stats.active_calls), tone: stats.active_calls > 0 ? "is-live" : "" },
        { label: "Failed", value: String(stats.failed_calls), tone: stats.failed_calls > 0 ? "is-alert" : "" },
        { label: "Errors · 24h", value: String(stats.errors_24h), tone: stats.errors_24h > 0 ? "is-alert" : "" },
        // Turn latency first: it is the silence the caller sits through.
        // Tool latency is kept but demoted — it runs in milliseconds, so
        // leading with it reports the system as fast while someone waits.
        { label: "Turn p95", value: `${(stats.p95_turn_latency_ms / 1000).toFixed(1)}s` },
        { label: "Turn mean", value: `${(stats.avg_turn_latency_ms / 1000).toFixed(1)}s` },
        { label: "Tool p95", value: `${stats.p95_tool_latency_ms}ms` },
      ]
    : [];

  return (
    <div className="figures">
      {items.map((item) => (
        <div className={`figure ${item.tone ?? ""}`} key={item.label}>
          <div className="value">{item.value}</div>
          <div className="label">{item.label}</div>
        </div>
      ))}
    </div>
  );
}

function CallDetail({ call, events }: { call: Call | null; events: CallEvent[] }) {
  if (!call) {
    return <div className="detail"><p className="state">Select a call to see what happened.</p></div>;
  }

  return (
    <div className="detail">
      <div className="detail-head">
        <h2>{who(call)}</h2>
        <div className="detail-meta">
          {/* Labelled, because it reads as the contact's number otherwise and
              is not: it is the number that rang, frozen at the time of the
              call. A caller who changes their number later still has this row
              showing the old one, which is the point. */}
          <span title="The number this call came from, as it was at the time">
            called from {phoneNumber(call)}
          </span>
          {numberChangedSince(call) ? (
            <span
              title="The contact's number in the CRM today. It has changed since this call."
              style={{ color: "var(--warn)" }}
            >
              CRM now {call.crm_phone}
            </span>
          ) : null}
          <span>{call.direction}</span>
          <span>{duration(call)}</span>
          {call.hubspot_contact_id ? <span>CRM {call.hubspot_contact_id}</span> : <span>no CRM match</span>}
          <span className={`tag ${call.status}`}>{call.status.replace("_", " ")}</span>
          <OutcomeChips outcomes={call.outcomes} />
        </div>
        <div className="detail-meta" style={{ color: "var(--ink-faint)" }}>
          <span>{call.call_id}</span>
        </div>
      </div>

      {call.summary ? (
        <div className="detail-summary">
          <span className="eyebrow">Summary written by the agent</span>
          {call.summary}
        </div>
      ) : null}

      {events.length === 0 ? (
        <p className="state">Nothing recorded on this call yet.</p>
      ) : (
        <div className="timeline">
          {events.map((event) => (
            <Turn key={event.id} event={event} />
          ))}
        </div>
      )}
    </div>
  );
}

function Turn({ event }: { event: CallEvent }) {
  const time = clockTime(event.created_at);

  if (event.kind === "user_speech" || event.kind === "agent_speech") {
    const isUser = event.kind === "user_speech";
    return (
      <div className={`turn speech ${isUser ? "user" : "agent"}`}>
        <div className="gutter">{time}</div>
        <div className="body">
          <div className="speaker">{isUser ? "Caller" : "Agent"}</div>
          <div className="said">{String(event.payload.text ?? "")}</div>
        </div>
      </div>
    );
  }

  // tool_call rows carry the args; the matching tool_result carries timing and
  // outcome. Rendering both keeps the causal order the agent actually followed.
  const isError =
    event.kind === "error" || event.payload.outcome === "user_error";
  const args = formatArgs(event.payload);
  const latency = event.latency_ms;

  return (
    <div className={`turn machine ${isError ? "is-error" : ""}`}>
      <div className="gutter">{time}</div>
      <div className="body">
        <span className="tool-name">
          {event.kind === "tool_call" ? "→" : isError ? "✕" : "←"} {event.name ?? event.kind}
        </span>
        {args ? <span className="args">{args}</span> : null}
        {isError && event.payload.reason ? (
          <span className="args">{String(event.payload.reason)}</span>
        ) : null}
        {latency != null ? (
          <span className={`latency ${latency > 1000 ? "slow" : ""}`}>
            <span className="bar" style={{ width: barWidth(latency) }} />
            {latency}ms
          </span>
        ) : null}
      </div>
    </div>
  );
}
