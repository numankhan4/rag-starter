"use client";

import { useState } from "react";
import { ingest, query, splitCitations, type QueryResponse } from "@/lib/api";

const SAMPLE = `Retrieval-augmented generation (RAG) answers questions using your own documents.
Documents are split into overlapping chunks, embedded as vectors and stored in a vector store.
At question time the most similar chunks are retrieved and the answer cites them as sources.`;

export default function Home() {
  const [docId, setDocId] = useState("intro");
  const [text, setText] = useState(SAMPLE);
  const [question, setQuestion] = useState("How are documents stored?");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [status, setStatus] = useState<string>("");
  const [busy, setBusy] = useState(false);

  async function run<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(true);
    setStatus(label);
    try {
      return await fn();
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Something went wrong");
      return undefined;
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <header>
        <h1>rag-starter</h1>
        <p>Ingest a document, then ask questions. Answers cite the chunks they came from.</p>
      </header>

      <section aria-labelledby="ingest-h">
        <h2 id="ingest-h">1. Ingest</h2>
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            const r = await run("Ingesting…", () => ingest(docId, text, docId));
            if (r) setStatus(`Ingested “${r.doc_id}” as ${r.chunks} chunk(s).`);
          }}
        >
          <label>
            Document id
            <input value={docId} onChange={(e) => setDocId(e.target.value)} required />
          </label>
          <label>
            Text
            <textarea value={text} onChange={(e) => setText(e.target.value)} rows={6} required />
          </label>
          <button disabled={busy}>Ingest</button>
        </form>
      </section>

      <section aria-labelledby="ask-h">
        <h2 id="ask-h">2. Ask</h2>
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            const r = await run("Searching…", () => query(question));
            if (r) {
              setResult(r);
              setStatus("");
            }
          }}
        >
          <label>
            Question
            <input value={question} onChange={(e) => setQuestion(e.target.value)} required />
          </label>
          <button disabled={busy}>Ask</button>
        </form>
      </section>

      <p role="status" aria-live="polite" className="status">
        {status}
      </p>

      {result && (
        <section aria-labelledby="answer-h" className="answer">
          <h2 id="answer-h">Answer</h2>
          <p>
            {splitCitations(result.answer).map((part, i) =>
              "cite" in part ? (
                <a key={i} href={`#source-${part.cite}`} className="cite">
                  [{part.cite}]
                </a>
              ) : (
                <span key={i}>{part.text}</span>
              ),
            )}
          </p>
          <h3>Sources</h3>
          <ol>
            {result.sources.map((s) => (
              <li key={s.n} id={`source-${s.n}`}>
                <strong>{s.title ?? s.doc_id}</strong> · chunk {s.chunk} · score {s.score.toFixed(3)}
                <blockquote>{s.text}</blockquote>
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}
