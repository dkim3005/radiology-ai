"use client";

import { useEffect, useState } from "react";

interface Study {
  id: string;
  patient_name: string;
  image_index: string;
}

interface Explain {
  study_id: string;
  top_class: string;
  predictions: { finding: string; score: number }[];
  cam_b64: string;
  explanation: string;
}

export default function ExplainabilityPage() {
  const [studies, setStudies] = useState<Study[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [data, setData] = useState<Explain | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/studies")
      .then((r) => r.json())
      .then((d: Study[]) => {
        setStudies(d);
        if (d.length) setSelected(d[3]?.id ?? d[0].id);
      })
      .catch(() => setError("Could not reach Radiology API"));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setData(null);
    setLoading(true);
    setError(null);
    fetch(`/api/studies/${selected}/explain`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "explain failed"))
      .finally(() => setLoading(false));
  }, [selected]);

  const study = studies.find((s) => s.id === selected);
  const maxScore = data ? Math.max(...data.predictions.map((p) => p.score)) : 1;

  return (
    <>
      <div className="page-header">
        <span className="eyebrow">Workflow · Explainability</span>
        <h1 className="page-title">Class Activation Maps</h1>
        <p className="page-sub">
          Which regions drove the classifier&apos;s prediction — a CAM heatmap
          derived from the DenseNet feature maps and classifier weights.
        </p>
      </div>

      {error && <div className="error-strip">{error}</div>}

      <div className="panel">
        <div className="panel-title">Study</div>
        <div className="chips">
          {studies.slice(0, 14).map((s) => (
            <button key={s.id}
              className={`chip${selected === s.id ? " chip" : ""}`}
              style={selected === s.id
                ? { borderColor: "var(--accent)", color: "var(--accent)" }
                : undefined}
              onClick={() => setSelected(s.id)}>
              {s.id}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="loading-row">
          <span className="spinner" /> Computing activation map…
        </div>
      )}

      {data && study && (
        <>
          <div className="panel">
            <div className="panel-title">
              {data.study_id} · {study.patient_name} · top class {data.top_class}
            </div>
            <div className="cam-grid">
              <div>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img className="cam-img" src={`/api/images/${study.image_index}`}
                  alt="original" />
                <div className="cam-caption">Original chest X-ray</div>
              </div>
              <div>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img className="cam-img"
                  src={`data:image/png;base64,${data.cam_b64}`}
                  alt="CAM overlay" />
                <div className="cam-caption">
                  CAM overlay — warmer = higher attention
                </div>
              </div>
            </div>
            <div className="explanation">{data.explanation}</div>
          </div>

          <div className="panel">
            <div className="panel-title">Top-5 Predictions</div>
            {data.predictions.map((p) => (
              <div className="finding-row" key={p.finding}>
                <span className="finding-name">{p.finding}</span>
                <div className="finding-track">
                  <div className="finding-fill"
                    style={{ width: `${(p.score / maxScore) * 100}%`,
                             background: "var(--accent)" }} />
                </div>
                <span className="finding-score">{p.score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="footer">
        <span>Radiology AI · Explainability</span>
        <span>FastAPI · PyTorch · Next.js · djkimlab.com</span>
      </div>
    </>
  );
}
