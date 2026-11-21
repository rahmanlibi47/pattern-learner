# Pattern Learner

A self-hosted, continuously running character-level learning lab. Start the Docker Compose stack once; the trainer periodically discovers licensed Wikipedia articles, stores source metadata and article text, incrementally trains versioned model candidates, evaluates them, and publishes immutable checkpoints. A separate FastAPI service serves generation requests from published models while training continues. A React + TypeScript GUI provides the dashboard, model registry, source inventory and prompt playground.

> **Scope:** this is an educational character-level model, not a modern general-purpose language model. The current baseline has 521 trainable parameters and is expected to produce limited, often incoherent text. “Self-training” here means autonomous source discovery and incremental training on collected text; it does not mean the model autonomously improves its own architecture or reliably learns from its own generated output.

## Start

Requirements: Docker Desktop with Docker Compose v2, and an internet connection for the initial image builds and Wikipedia collection.

From the repository root:

```powershell
Copy-Item .env.example .env
docker compose up -d --build
```

Open:

- GUI: <http://localhost:3000>
- API health: <http://localhost:8000/health>
- API docs: <http://localhost:8000/docs>

The trainer begins discovering articles automatically. Initial collection/training may take several minutes. Follow its logs:

```powershell
docker compose logs -f trainer
```

Other useful commands:

```powershell
docker compose ps
docker compose logs -f api
docker compose logs -f gui
docker compose down
```

`docker compose down` stops/removes containers but preserves the named `pattern-data` volume. **Do not use `docker compose down -v` unless you intend to delete the model registry, article corpus and checkpoints.**

## What runs in Docker

| Service | Role | Port |
|---|---|---:|
| `trainer` | Serial discovery/training worker; never handles generation requests | internal |
| `api` | FastAPI, SQLite metadata, model registry, SSE events and inference | 8000 |
| `gui` | Production Vite build served by Nginx; proxies API and SSE requests | 3000 |

The API and trainer share the same persistent named volume at `/app/storage`. SQLite uses WAL mode and a busy timeout to tolerate concurrent API/worker access. Docker Compose waits for the API health check before starting the trainer and GUI.

## Independent training and generation

- The trainer runs as its own process/container. It does not run inside an API request.
- Generation selects a model with `status=ready`; a candidate in `training` state is never selectable.
- A candidate is initialized from the latest ready model (incremental learning) and trained against the cumulative locally stored training split. This replay of older training material helps reduce forgetting when new articles are added.
- Each article receives a persisted train/validation/test assignment when first stored. Existing assignments are not changed when later articles arrive, preventing validation/test articles from silently leaking into subsequent training runs.
- Candidate weights are written into a temporary directory. A candidate is promoted only if its validation loss is within the configured regression tolerance of its parent evaluated on the same current validation split. After checkpoint writing, its directory is atomically renamed and the database record is marked `ready`. Rejected candidates retain metrics/metadata but no selectable checkpoint; published checkpoints are never overwritten.
- Candidates that fail technically are marked `failed`; candidates that exceed the validation-regression tolerance are marked `rejected`. In both cases, the last ready model remains available.
- Only one candidate trains at a time. Generation is served independently by the API and can use any ready version while the next candidate trains.
- If discovery finds no new articles in the training split, the trainer waits until the next discovery interval rather than generating meaningless model versions from identical training data. New validation/test-only articles are recorded but do not trigger a redundant training cycle.

## GUI

- **Overview:** worker state, current epoch/progress, live metrics, model versions and recent events.
- **Model registry:** ready, training, rejected and failed versions, parent lineage, parameter count and evaluation metrics.
- **Data sources:** article title, topic, persisted dataset split, character count, source URL, reported license and collection time. Filter the inventory to a specific model version to inspect its source set.
- **Prompt playground:** select a ready model, enter a prompt, set temperature and output length, and generate independently of training.

The browser receives server-sent events (SSE) and also refreshes status periodically. The prompt playground uses a normal HTTP request to the API.

## Configuration

Copy `.env.example` to `.env` and adjust values before starting the stack. Main settings:

| Variable | Default | Meaning |
|---|---:|---|
| `DISCOVERY_INTERVAL_SECONDS` | `1800` | Wait between discovery cycles (30 minutes) |
| `TRAINING_EPOCHS` | `20` | Epochs per candidate |
| `BATCH_SIZE` | `128` | Mini-batch size |
| `LEARNING_RATE` | `0.003` | Adam learning rate |
| `CONTEXT_SIZE` | `8` | Number of preceding characters |
| `EMBEDDING_DIM` | `2` | Character embedding width |
| `HIDDEN_SIZE` | `8` | Hidden-layer width |
| `MAX_PARAMETERS` | `1000` | Hard parameter-count ceiling for candidate models |
| `PROMOTION_MAX_REGRESSION` | `0.02` | Maximum allowed validation-loss regression versus parent for promotion |
| `WIKIPEDIA_USER_AGENT` | project default | User-Agent sent to Wikimedia |

Changing architecture dimensions after models have been published is not yet a supported migration. Use a fresh storage volume for a different architecture, or implement an explicit model-version compatibility/migration mechanism first.

## Data and licensing

The current collector uses the Wikipedia REST API and accepts articles whose API-reported license is Creative Commons Attribution-ShareAlike 3.0 or 4.0. It skips obvious disambiguation pages, journal pages, albums, songs, films and other unrelated search results. The GUI retains source URLs, reported license metadata and each article's persisted dataset split for attribution and traceability. The source page can be filtered to the source set associated with a specific model version. A page-level API license is not a guarantee that every embedded/third-party asset on a page has the same license; this prototype trains on extracted article text, not media assets.

Article text is stored in the persistent local database so that later incremental candidates can replay prior material and experiments remain traceable. It is not copied into the source repository. The current corpus is English-only and normalized to lowercase letters plus basic punctuation.

## API endpoints

- `GET /health`
- `GET /api/status`
- `GET /api/models`
- `GET /api/sources`
- `GET /api/events` (SSE)
- `POST /api/generate`

Example generation request:

```json
{
  "model_id": "model-0001",
  "prompt": "the science",
  "max_new_tokens": 120,
  "temperature": 0.8,
  "seed": 42
}
```

Temperature `0` uses greedy decoding. A positive temperature samples from the model's character distribution. The prompt must contain at least the model's context length and use characters in its vocabulary.

## Development and tests

Backend tests can be run locally with Python 3.12:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

Frontend development (outside Docker):

```powershell
# Terminal 1, from repository root:
python -m pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000

# Terminal 2, from repository root:
python -m src.training.worker

# Terminal 3:
cd gui
npm install
npm run dev
```

For normal use, use Docker Compose. The GUI container is a production build, not a development server.

## Current limitations

- Small character-level model; generated prose quality is low by design.
- One trainer process and one candidate at a time.
- The collector's search relevance rules are heuristic, not semantic classification.
- New model candidates are created only when genuinely new licensed articles are collected.
- Incremental training resets Adam's optimizer moments at each model version.
- The SQLite database and shared volume are intended for one local host, not multiple machines or high write concurrency.
- No authentication is enabled; keep the exposed ports on a trusted local network. Do not expose this prototype directly to the public internet.
