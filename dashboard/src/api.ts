import type { Call, CallEvent, ErrorRow, Stats } from "./types";

/** Thrown for any non-2xx so callers can show the status, not a blank screen. */
export class ApiError extends Error {
  // Declared and assigned explicitly rather than as a constructor parameter
  // property: Vite's tsconfig sets `erasableSyntaxOnly`, which rejects the
  // shorthand because it emits runtime code.
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function get<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch {
    // fetch only rejects on network failure, which for this UI means the API
    // process is not running — worth saying plainly rather than "failed".
    throw new ApiError("Can't reach the API. Is the server running?", 0);
  }
  if (!response.ok) {
    throw new ApiError(`Request failed (${response.status})`, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  stats: () => get<Stats>("/api/stats"),
  calls: () => get<Call[]>("/api/calls?limit=100"),
  call: (id: string) => get<Call>(`/api/calls/${encodeURIComponent(id)}`),
  events: (id: string) =>
    get<CallEvent[]>(`/api/calls/${encodeURIComponent(id)}/events`),
  errors: () => get<ErrorRow[]>("/api/errors"),
};
