# inboxpilot-lite

Production-quality starter API for classifying incoming messages with a local LM Studio model and a rules-based fallback.

## Features
- FastAPI endpoints: `POST /classify`, `POST /classify/batch`, `GET /health`, `GET /metrics`
- Categories: `question | complaint | sales | spam | other`
- Request ID middleware with `x-request-id` propagation
- In-memory per-IP rate limiting
- In-memory metrics counters via `/metrics`
- Uses the OpenAI Python client against LM Studio's OpenAI-compatible API
- Classifier selection:
  - Uses `LMStudioClassifier` (OpenAI-compatible local API)
  - Falls back to `RulesClassifier` if LM Studio is unreachable or returns invalid output
- CI via GitHub Actions (`pytest` + `ruff`)
- Pre-commit hooks (`ruff`, `ruff-format`)

## Project structure
```text
app/
  core/config.py
  core/metrics.py
  core/rate_limit.py
  main.py
  models/schemas.py
  services/classifier.py
  services/lmstudio_classifier.py
tests/
  test_classify.py
```

## Setup
1. Create and activate a virtual environment:
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
macOS/Linux:
```bash
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
This installs the OpenAI Python client used for LM Studio requests.

3. Configure environment:
```bash
cp .env.example .env
```

## Environment variables
- `LOG_LEVEL` (default: `INFO`)
- `LMSTUDIO_BASE_URL` (default: `http://localhost:1234/v1`)
- `LMSTUDIO_API_KEY` (default: `lm-studio`)
- `LMSTUDIO_MODEL` (default: `openai/gpt-oss-20b`)
- `LMSTUDIO_TIMEOUT_SECONDS` (default: `20`)
- `RATE_LIMIT_REQUESTS` (default: `60`)
- `RATE_LIMIT_WINDOW_SECONDS` (default: `60`)
- `MAX_BATCH_SIZE` (default: `20`)

For LM Studio:
```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=openai/gpt-oss-20b
```

If LM Studio is not reachable, the app logs a warning and falls back to `RulesClassifier`.

## Architecture
```text
Client
  |
  v
FastAPI app (app/main.py)
  |
  +--> Middleware
  |      - request_id context + response header
  |      - per-IP rate limit
  |      - counters + structured logs
  |
  +--> Routes
         - GET /health
         - GET /metrics
         - POST /classify
         - POST /classify/batch
                |
                v
          Classifier services
                |
                +--> LMStudioClassifier (app/services/lmstudio_classifier.py)
                |
                +--> RulesClassifier (fallback)
```

## Run locally
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run tests
```bash
pytest -q
```

## API examples
Health:
```bash
curl http://localhost:8000/health
```

Classify:
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"How do I reset my password?\"}"
```

Batch classify:
```bash
curl -X POST http://localhost:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d "{\"texts\":[\"How can I upgrade?\",\"I need a refund\"]}"
```

Example classify response:
```json
{
  "category": "question",
  "confidence": 0.84,
  "suggested_reply": "Thanks for your question. Share a bit more detail and I can help quickly."
}
```

## Tooling
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Roadmap
1. Move metrics/rate-limit state to Redis for multi-instance deployments.
2. Add OpenTelemetry tracing and latency histograms.
3. Add auth and tenant-aware quotas.
4. Add async queue support for very large batch jobs.
