"use client";

import { useEffect, useState } from "react";

interface Study {
  id: string;
  mrn: string;
  patient_name: string;
  patient_age: number;
  patient_gender: string;
  view_position: string;
  study_date: string;
  image_index: string;
  finding_labels: string;
  status: string;
}

interface Finding {
  finding: string;
  score: number;
  confidence: "High" | "Moderate" | "Low";
}

interface Infer {
  primary: Finding;
  supporting: Finding[];
  all: Finding[];
}

interface FhirRes {
  resource: string;
  fhir_id: string;
  content: string;
}

const CONF_COLOR: Record<string, string> = {
  High: "#f87171", Moderate: "#fbbf24", Low: "#6b8499",
};

export default function TriagePage() {
  const [studies, setStudies] = useState<Study[]>([]);
  const [selected, setSelected] = useState<Study | null>(null);
  const [infer, setInfer] = useState<Infer | null>(null);
  const [fhir, setFhir] = useState<FhirRes[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/studies")
      .then((r) => r.json())
      .then((d: Study[]) => {
        setStudies(d);
        if (d.length) setSelected(d[0]);
      })
      .catch(() => setError("Could not reach Radiology API"));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setInfer(null);
    setFhir([]);
    setLoading(true);
    Promise.all([
      fetch(`/api/studies/${selected.id}/infer`).then((r) => r.json()),
      fetch(`/api/studies/${selected.id}/fhir`).then((r) => r.json()),
    ])
      .then(([inf, fh]) => {
        setInfer(inf);
        setFhir(fh.resources ?? []);
      })
      .catch(() => setError("Inference failed"))
      .finally(() => setLoading(false));
  }, [selected]);

  return (
    <>
      <div className="page-header">
        <span className="eyebrow">Workflow · Triage</span>
        <h1 className="page-title">Radiology Worklist</h1>
        <p className="page-sub">
          DenseNet121 multi-label classification over the NIH ChestX-ray14
          corpus, with FHIR-style patient and imaging context.
        </p>
      </div>

      {error && <div className="error-strip">{error}</div>}

      <div className="split">
        <div className="panel">
          <div className="panel-title">Worklist — {studies.length} studies</div>
          <div className="worklist">
            {studies.map((s) => (
              <div key={s.id}
                className={`study-row${selected?.id === s.id ? " active" : ""}`}
                onClick={() => setSelected(s)}>
                <div className="study-id">{s.id}</div>
                <div className="study-name">{s.patient_name}</div>
                <div className="study-meta">
                  {s.patient_age}{s.patient_gender} · {s.view_position} · {s.study_date}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          {selected && (
            <>
              <div className="panel">
                <div className="panel-title">
                  {selected.id} · {selected.patient_name} · {selected.mrn}
                </div>
                <div className="xray-wrap">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`/api/images/${selected.image_index}`} alt={selected.id} />
                </div>
              </div>

              <div className="panel">
                <div className="panel-title">AI Findings</div>
                {loading && (
                  <div className="loading-row">
                    <span className="spinner" /> Running DenseNet inference…
                  </div>
                )}
                {infer && (
                  <>
                    <div className="primary-card">
                      <div className="primary-finding">
                        {infer.primary.finding}{" "}
                        <span className={`badge badge-${infer.primary.confidence}`}>
                          {infer.primary.confidence}
                        </span>
                      </div>
                      <div className="primary-meta">
                        primary finding · model score {infer.primary.score.toFixed(3)}
                      </div>
                    </div>
                    {infer.all.slice(0, 8).map((f) => (
                      <div className="finding-row" key={f.finding}>
                        <span className="finding-name">{f.finding}</span>
                        <div className="finding-track">
                          <div className="finding-fill"
                            style={{ width: `${Math.min(100, f.score * 100)}%`,
                                     background: CONF_COLOR[f.confidence] }} />
                        </div>
                        <span className="finding-score"
                          style={{ color: CONF_COLOR[f.confidence] }}>
                          {f.score.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </>
                )}
              </div>

              {fhir.length > 0 && (
                <div className="panel">
                  <div className="panel-title">FHIR Context</div>
                  {fhir.map((r) => (
                    <div className="fhir-row" key={r.fhir_id}>
                      <span className="fhir-res">{r.resource}</span>
                      <span className="fhir-id">{r.fhir_id}</span>
                      <span>{r.content}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <div className="footer">
        <span>Radiology AI · Triage</span>
        <span>FastAPI · PyTorch · Next.js · djkimlab.com</span>
      </div>
    </>
  );
}
