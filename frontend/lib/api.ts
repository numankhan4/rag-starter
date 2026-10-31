export interface Source {
  n: number;
  doc_id: string;
  title: string | null;
  chunk: number;
  score: number;
  text: string;
}

export interface QueryResponse {
  answer: string;
  citations: number[];
  sources: Source[];
}

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function post<T>(path: string, body: unknown, base = API_URL, fetchImpl: typeof fetch = fetch): Promise<T> {
  const res = await fetchImpl(`${base}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `Request failed with ${res.status}`;
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* non-JSON error */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export function ingest(docId: string, text: string, title?: string, base?: string, fetchImpl?: typeof fetch) {
  return post<{ doc_id: string; chunks: number }>("/ingest", { doc_id: docId, text, title }, base, fetchImpl);
}

export function query(question: string, k = 4, base?: string, fetchImpl?: typeof fetch) {
  return post<QueryResponse>("/query", { question, k }, base, fetchImpl);
}

/** Split an answer like "Foo [1] bar [2]" into text and citation parts for rendering. */
export function splitCitations(answer: string): Array<{ text: string } | { cite: number }> {
  const parts: Array<{ text: string } | { cite: number }> = [];
  const re = /\[(\d+)\]/g;
  let last = 0;
  for (const m of answer.matchAll(re)) {
    const i = m.index ?? 0;
    if (i > last) parts.push({ text: answer.slice(last, i) });
    parts.push({ cite: Number(m[1]) });
    last = i + m[0].length;
  }
  if (last < answer.length) parts.push({ text: answer.slice(last) });
  return parts;
}
