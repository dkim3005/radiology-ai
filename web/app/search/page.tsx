"use client";

import { useCallback, useRef, useState } from "react";

interface Result {
  filename: string;
  score: number;
  rank: number;
}

interface Tag {
  tag: string;
  score: number;
}

const EXAMPLES = ["cardiomegaly", "pleural effusion", "normal chest", "pneumonia"];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [tags, setTags] = useState<Tag[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"text" | "image">("text");
  const fileRef = useRef<HTMLInputElement>(null);

  const loadTags = useCallback(async (filename: string) => {
    try {
      const r = await fetch(`/api/search/tags/${encodeURIComponent(filename)}`);
      if (r.ok) setTags(await r.json());
    } catch {
      /* ignore */
    }
  }, []);

  const textSearch = useCallback(async (q: string) => {
    const t = q.trim();
    if (!t) return;
    setLoading(true);
    setError(null);
    setTags(null);
    try {
      const r = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: t, top_k: 8 }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data: Result[] = await r.json();
      setResults(data);
      if (data.length) loadTags(data[0].filename);
    } catch (e) {
      setError(e instanceof Error ? e.message : "search failed");
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [loadTags]);

  const imageSearch = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setTags(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch("/api/search/image?top_k=8", {
        method: "POST",
        body: form,
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data: Result[] = await r.json();
      setResults(data);
      if (data.length) loadTags(data[0].filename);
    } catch (e) {
      setError(e instanceof Error ? e.message : "search failed");
    } finally {
      setLoading(false);
    }
  }, [loadTags]);

  const maxTag = tags && tags.length ? tags[0].score : 1;

  return (
    <>
      <div className="page-header">
        <span className="eyebrow">Workflow · Semantic Search</span>
        <h1 className="page-title">CLIP Visual Search</h1>
        <p className="page-sub">
          Zero-shot semantic search over the X-ray corpus with OpenAI CLIP
          ViT-B/32 — query by text or by uploading an image.
        </p>
      </div>

      <div className="panel">
        <div className="chips">
          <button className="chip"
            style={mode === "text" ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}
            onClick={() => setMode("text")}>Text query</button>
          <button className="chip"
            style={mode === "image" ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}
            onClick={() => setMode("image")}>Image query</button>
        </div>

        {mode === "text" ? (
          <>
            <div className="search-row">
              <input value={query} placeholder="Describe a finding…"
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && textSearch(query)} />
              <button className="btn" onClick={() => textSearch(query)}
                disabled={loading || !query.trim()}>Search</button>
            </div>
            <div className="chips">
              {EXAMPLES.map((q) => (
                <button key={q} className="chip"
                  onClick={() => { setQuery(q); textSearch(q); }}>{q}</button>
              ))}
            </div>
          </>
        ) : (
          <div className="drop-zone" onClick={() => fileRef.current?.click()}>
            Click to upload a chest X-ray — finds visually similar studies
            <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }}
              onChange={(e) => e.target.files?.[0] && imageSearch(e.target.files[0])} />
          </div>
        )}
      </div>

      {error && <div className="error-strip">{error}</div>}
      {loading && (
        <div className="loading-row"><span className="spinner" /> Running CLIP encoder…</div>
      )}

      {results.length > 0 && !loading && (
        <div className="panel">
          <div className="panel-title">Results — {results.length} images</div>
          <div className="results-grid">
            {results.map((r) => (
              <div className="result-card" key={r.filename}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`/api/images/${encodeURIComponent(r.filename)}`}
                  alt={r.filename} loading="lazy" />
                <div className="result-meta">
                  <span className="result-rank">#{r.rank}</span>
                  <span className="result-score">{r.score.toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {tags && !loading && (
        <div className="panel">
          <div className="panel-title">CLIP Tags — top result</div>
          {tags.map((t) => (
            <div className="tag-row" key={t.tag}>
              <span className="tag-name">{t.tag}</span>
              <div className="tag-track">
                <div className="tag-fill"
                  style={{ width: `${Math.max(0, (t.score / maxTag) * 100)}%` }} />
              </div>
              <span className="tag-score">{t.score.toFixed(3)}</span>
            </div>
          ))}
        </div>
      )}

      {results.length === 0 && !loading && !error && (
        <div className="empty">Enter a query to search the chest X-ray corpus.</div>
      )}

      <div className="footer">
        <span>Radiology AI · Semantic Search</span>
        <span>FastAPI · CLIP · Next.js · djkimlab.com</span>
      </div>
    </>
  );
}
