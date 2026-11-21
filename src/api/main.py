from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.database import store
from src.generation.engine import generate_text

@asynccontextmanager
async def lifespan(_app: FastAPI):
    store.init_db()
    yield


app = FastAPI(title="Pattern Learner API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {**store.get_runtime(), "counts": store.counts()}


@app.get("/api/models")
def models() -> dict[str, Any]:
    items = []
    for model in store.list_models():
        model.pop("checkpoint_path", None)
        items.append(model)
    return {"items": items}


@app.get("/api/sources")
def sources(limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
    return {"items": store.list_articles(limit)}


@app.get("/api/events")
async def events(request: Request, after: int = 0) -> StreamingResponse:
    async def stream():
        last_event_header = request.headers.get("last-event-id")
        if last_event_header:
            cursor = max(after, int(last_event_header))
        elif after:
            cursor = after
        else:
            # On a fresh browser visit, send only a recent activity window.
            cursor = max(0, store.latest_event_id() - 50)
        while True:
            records = store.get_events(cursor, limit=100)
            if records:
                for record in records:
                    cursor = record["id"]
                    yield f"id: {cursor}\nevent: {record['type']}\ndata: {json.dumps(record['data'])}\n\n"
            else:
                yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class GenerateRequest(BaseModel):
    model_id: str
    prompt: str = Field(min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=120, ge=1, le=500)
    temperature: float = Field(default=0.8, ge=0, le=2.0)
    seed: int | None = None



@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    record = store.get_model(request.model_id)
    if not record or record["status"] != "ready" or not record.get("checkpoint_path"):
        raise HTTPException(status_code=404, detail="That model is not ready or does not exist.")
    prompt = request.prompt.lower()
    try:
        result = generate_text(
            request.model_id,
            record["checkpoint_path"],
            prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            seed=request.seed,
        )
        return {
            "model_id": request.model_id,
            "prompt": prompt,
            "generated_text": result,
            "new_characters": len(result) - len(prompt),
            "temperature": request.temperature,
            "seed": request.seed,
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Generation failed: {error}") from error
