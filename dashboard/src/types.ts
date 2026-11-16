export type CallStatus = "in_progress" | "completed" | "failed";

export interface Call {
  call_id: string;
  direction: "inbound" | "outbound";
  from_number: string | null;
  to_number: string | null;
  hubspot_contact_id: string | null;
  /** Who they were on this call. A snapshot, not a live CRM lookup. */
  caller_name: string | null;
  status: CallStatus;
  summary: string | null;
  started_at: string;
  ended_at: string | null;
  /** What the call produced: orders placed, opportunities opened, bookings. */
  outcomes: Outcome[];
  /** What the CRM holds *now*, read live. Null when the contact is unknown,
   *  deleted, or HubSpot could not be reached — the page still renders. */
  crm_phone: string | null;
  crm_name: string | null;
}

/** A business result recorded on the timeline, not just a thing that was said. */
export interface Outcome {
  kind: "order_placed" | "purchase_interest" | "appointment_booked";
  order_ref?: string;
  sku?: string;
  quantity?: number;
  deal_id?: string | null;
  amount_cents?: number;
  [key: string]: unknown;
}

export type EventKind =
  | "user_speech"
  | "agent_speech"
  | "tool_call"
  | "tool_result"
  | "error";

export interface CallEvent {
  id: number;
  kind: EventKind;
  name: string | null;
  payload: Record<string, unknown>;
  latency_ms: number | null;
  created_at: string;
}

export interface Stats {
  total_calls: number;
  active_calls: number;
  failed_calls: number;
  calls_24h: number;
  errors_24h: number;
  /** What the caller experiences: their silence between speaking and hearing. */
  avg_turn_latency_ms: number;
  p95_turn_latency_ms: number;
  /** Useful for spotting a slow integration, but never the headline: tool
   *  execution runs in milliseconds while turns run in seconds. */
  avg_tool_latency_ms: number;
  p95_tool_latency_ms: number;
}

export interface ErrorRow {
  id: number;
  call_id: string;
  name: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}
