import { test } from "node:test";
import assert from "node:assert/strict";
import { query, splitCitations } from "./api.ts";

test("splitCitations separates text and numbered citations", () => {
  assert.deepEqual(splitCitations("FastAPI uses Pydantic [1]. Next.js is React [2]"), [
    { text: "FastAPI uses Pydantic " },
    { cite: 1 },
    { text: ". Next.js is React " },
    { cite: 2 },
  ]);
  assert.deepEqual(splitCitations("no cites"), [{ text: "no cites" }]);
});

test("query posts JSON and surfaces API error details", async () => {
  const calls: Array<{ url: string; body: string }> = [];
  const ok = (async (url: string, init?: RequestInit) => {
    calls.push({ url, body: String(init?.body) });
    return new Response(JSON.stringify({ answer: "a [1]", citations: [1], sources: [] }), { status: 200 });
  }) as typeof fetch;
  const res = await query("what?", 3, "http://api", ok);
  assert.equal(calls[0]?.url, "http://api/query");
  assert.deepEqual(JSON.parse(calls[0]!.body), { question: "what?", k: 3 });
  assert.equal(res.answer, "a [1]");

  const bad = (async () => new Response(JSON.stringify({ detail: "Document has no text" }), { status: 422 })) as typeof fetch;
  await assert.rejects(query("x", 4, "http://api", bad), /Document has no text/);
});
