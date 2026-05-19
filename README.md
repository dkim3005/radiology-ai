# Radiology AI

A chest X-ray analysis platform over the NIH ChestX-ray14 corpus. Three
workflows, one shared DenseNet — the merge of three earlier projects (Medical
AI, Explainable AI, CLIP Visual Search) into one coherent system.

## Workflows

- **Triage** — DenseNet121 multi-label classification with a radiology
  worklist and FHIR-style patient/imaging context.
- **Explainability** — Class Activation Maps showing which image regions drove
  each prediction, derived from the classifier's feature maps and weights.
- **Semantic Search** — zero-shot text-to-image and image-to-image retrieval
  with OpenAI CLIP ViT-B/32, plus auto-generated CLIP tags.

The DenseNet model is loaded once and shared by triage and explainability;
CLIP is loaded for search.

## Stack

- Backend: FastAPI (port 10001) — torchxrayvision DenseNet121 + open_clip
- Frontend: Next.js 15 (port 10000)

## Run

```bash
python3 api.py
cd web && npm install && npm run build && npm start
```

Open http://localhost:10000

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/studies` | Worklist of studies |
| GET | `/api/studies/{id}/infer` | DenseNet classification |
| GET | `/api/studies/{id}/explain` | CAM heatmap + predictions |
| GET | `/api/studies/{id}/fhir` | FHIR resource context |
| GET | `/api/images/{name}` | Serve an X-ray image |
| POST | `/api/search/text` | CLIP text-to-image search |
| POST | `/api/search/image` | CLIP image-to-image search |
| GET | `/api/search/tags/{file}` | CLIP semantic tags |
