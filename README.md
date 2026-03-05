# inboxpilot-lite

Production-quality starter API for classifying incoming messages with an optional OpenAI path and a guaranteed rules-based fallback.

## Features
- FastAPI service with two endpoints: `POST /classify`, `GET /health`
- Categories: `question | complaint | sales | spam | other`
- Confidence score from `0` to `1`
- Suggested short helpful reply
- Classifier selection:
  - Uses `OpenAIClassifier` when `OPENAI_API_KEY` is configured
  - Falls back to `RulesClassifier` otherwise (or if OpenAI call fails)
- OpenAI path safety:
  - Strict JSON schema contract requested from model
  - Pydantic validation for model output
  - Per-request timeout
  - Retry with exponential backoff for transient failures
- Structured-ish logging with timestamp, level, and `request_id`
- Tests with `pytest` + `httpx`

## Project structure
```text
app/
  core/config.py
  main.py
  models/schemas.py
  services/classifier.py
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

3. Configure environment:
```bash
cp .env.example .env
```
Set `OPENAI_API_KEY` only if you want LLM-backed classification.

### Environment variables
- `LOG_LEVEL` (default: `INFO`)
- `OPENAI_API_KEY` (optional; if empty, rules-based classifier is used)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_TIMEOUT_SECONDS` (default: `8`)
- `OPENAI_MAX_RETRIES` (default: `2`)
- `OPENAI_RETRY_BACKOFF_SECONDS` (default: `0.4`)

## Run locally
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or with Make:
```bash
make run
```

## Run tests
```bash
pytest -q
```

Or with Make:
```bash
make test
```

## Docker
Build and run with compose:
```bash
docker compose up --build
```

Direct Docker:
```bash
docker build -t inboxpilot-lite .
docker run --rm -p 8000:8000 --env-file .env inboxpilot-lite
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

Example response:
```json
{
  "category": "question",
  "confidence": 0.84,
  "suggested_reply": "Thanks for your question. Share a bit more detail and I can help quickly."
}
```

Another valid response:
```json
{
  "category": "complaint",
  "confidence": 0.93,
  "suggested_reply": "I'm sorry about this. Please share your order number so we can fix it quickly."
}
```

## Add new categories
1. Update category literal in `app/models/schemas.py` (`Category`).
2. Update `VALID_CATEGORIES` and rule logic in `app/services/classifier.py`.
3. If using OpenAI, update the prompt in `OpenAIClassifier._build_payload`.
4. Add tests in `tests/test_classify.py` for the new category behavior.
