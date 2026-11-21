from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from src.data.collector import collect_articles
from src.data.prepare import normalize_text
from src.dataset import create_dataset
from src.database import store
from src.loss import cross_entropy_loss
from src.model import PatternLearner
from src.optimizer import Adam

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/app/storage"))
MODELS_DIR = STORAGE_DIR / "models"
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "8"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2"))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "8"))
EPOCHS = int(os.getenv("TRAINING_EPOCHS", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.003"))
POLL_SECONDS = int(os.getenv("DISCOVERY_INTERVAL_SECONDS", "1800"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
MAX_PARAMETERS = int(os.getenv("MAX_PARAMETERS", "1000"))
PROMOTION_MAX_REGRESSION = float(os.getenv("PROMOTION_MAX_REGRESSION", "0.02"))
ALPHABET = "abcdefghijklmnopqrstuvwxyz .,!?;:'-"


def log(message: str, level: str = "info") -> None:
    print(f"[{level.upper()}] {message}", flush=True)
    store.emit("log", {"level": level, "message": message})


def split_articles(articles: list[dict]) -> dict[str, list[dict]]:
    """Use persisted article-level assignments; never leak prior eval data into training."""
    if len(articles) < 3:
        raise ValueError("At least three distinct licensed articles are required.")
    if all(article.get("split") in {"train", "validation", "test"} for article in articles):
        splits = {
            name: [article for article in articles if article.get("split") == name]
            for name in ("train", "validation", "test")
        }
        if all(splits.values()):
            return splits
    # Fallback for unit tests or older article records without persisted splits.
    ordered = sorted(articles, key=lambda a: hashlib.sha256(a["id"].encode()).hexdigest())
    n = len(ordered)
    train_n = max(1, int(n * 0.70))
    val_n = max(1, int(n * 0.15 + 0.5))
    if train_n + val_n >= n:
        train_n, val_n = n - 2, 1
    return {
        "train": ordered[:train_n],
        "validation": ordered[train_n:train_n + val_n],
        "test": ordered[train_n + val_n:],
    }


def build_examples(articles: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    char_to_id = {char: i for i, char in enumerate(ALPHABET)}
    for article in articles:
        text = normalize_text(article["text"])
        # The normalizer and fixed alphabet must stay in sync.
        unknown = set(text) - set(char_to_id)
        if unknown:
            raise ValueError(f"Normalizer emitted unsupported characters: {unknown}")
        tokens = [char_to_id[c] for c in text]
        x, y = create_dataset(tokens, CONTEXT_SIZE)
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("The split did not produce any character examples.")
    return np.concatenate(xs), np.concatenate(ys)


def evaluate(model: PatternLearner, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    loss_sum = 0.0
    correct = 0
    total = len(y)
    for start in range(0, total, 512):
        bx, by = x[start:start + 512], y[start:start + 512]
        logits = model.forward(bx)
        loss_sum += cross_entropy_loss(logits, by) * len(by)
        correct += int(np.sum(np.argmax(logits, axis=1) == by))
    return loss_sum / total, correct / total * 100


def should_promote(parent_validation_loss: float | None, candidate_validation_loss: float) -> bool:
    if not np.isfinite(candidate_validation_loss):
        return False
    if parent_validation_loss is None:
        return True
    return candidate_validation_loss <= parent_validation_loss + PROMOTION_MAX_REGRESSION


def load_parent(model: PatternLearner, parent: dict | None) -> None:
    if not parent or not parent.get("checkpoint_path"):
        return
    with np.load(parent["checkpoint_path"], allow_pickle=False) as checkpoint:
        model.E = checkpoint["E"].copy()
        model.W1 = checkpoint["W1"].copy()
        model.b1 = checkpoint["b1"].copy()
        model.W2 = checkpoint["W2"].copy()
        model.b2 = checkpoint["b2"].copy()


def next_model_id() -> str:
    existing = store.list_models()
    number = len(existing) + 1
    return f"model-{number:04d}"


def dataset_fingerprint(articles: list[dict]) -> str:
    # Only the training split determines whether the model has new material
    # to learn. New validation/test articles should not trigger fake retraining.
    training_articles = [article for article in articles if article.get("split") == "train"]
    if not training_articles:
        training_articles = articles
    source_ids = [article["id"] for article in training_articles]
    return hashlib.sha256("".join(sorted(source_ids)).encode()).hexdigest()


def train_candidate(articles: list[dict], parent: dict | None) -> None:
    splits = split_articles(articles)
    train_x, train_y = build_examples(splits["train"])
    val_x, val_y = build_examples(splits["validation"])
    test_x, test_y = build_examples(splits["test"])

    source_ids = [article["id"] for article in articles]
    dataset_hash = dataset_fingerprint(splits["train"])
    model_id = next_model_id()
    model = PatternLearner(
        vocab_size=len(ALPHABET), context_size=CONTEXT_SIZE,
        embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE,
        seed=RANDOM_SEED,
    )
    load_parent(model, parent)
    parent_validation_loss = None
    if parent is not None:
        parent_validation_loss, _ = evaluate(model, val_x, val_y)
    parameter_count = model.count_parameters()
    if parameter_count > MAX_PARAMETERS:
        raise ValueError(
            f"Configured model has {parameter_count} parameters; limit is {MAX_PARAMETERS}."
        )
    optimizer = Adam(learning_rate=LEARNING_RATE)
    parameters = {name: getattr(model, name) for name in ("E", "W1", "b1", "W2", "b2")}

    store.create_model(model_id, parent["id"] if parent else None, EPOCHS, source_ids, dataset_hash)
    store.update_model(model_id, parameter_count=parameter_count)
    store.set_runtime(
        status="training", stage="training", current_model_id=model_id,
        current_epoch=0, total_epochs=EPOCHS, progress=0,
        message=f"Training {model_id}", train_examples=len(train_y),
        validation_examples=len(val_y), test_examples=len(test_y),
    )
    store.emit("model", {"id": model_id, "status": "training"})
    log(f"Started {model_id}; parent={parent['id'] if parent else 'none'}; articles={len(articles)}; parameters={parameter_count}")

    rng = np.random.default_rng(RANDOM_SEED)
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        indices = rng.permutation(len(train_x))
        for start in range(0, len(train_x), BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            bx, by = train_x[batch_idx], train_y[batch_idx]
            logits = model.forward(bx)
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(shifted)
            probs /= np.sum(probs, axis=1, keepdims=True)
            dlogits = probs
            dlogits[np.arange(len(by)), by] -= 1
            dlogits /= len(by)
            gradients = model.backward(bx, dlogits)
            optimizer.step(parameters, gradients)

        train_loss, train_acc = evaluate(model, train_x, train_y)
        val_loss, val_acc = evaluate(model, val_x, val_y)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {name: value.copy() for name, value in parameters.items()}

        store.update_model(
            model_id, epoch=epoch, train_loss=train_loss,
            validation_loss=val_loss, train_accuracy=train_acc,
            validation_accuracy=val_acc,
        )
        store.set_runtime(
            status="training", stage="training", current_model_id=model_id,
            current_epoch=epoch, total_epochs=EPOCHS,
            progress=round(epoch / EPOCHS * 100, 1),
            train_loss=train_loss, validation_loss=val_loss,
            train_accuracy=train_acc, validation_accuracy=val_acc,
            message=f"{model_id}: epoch {epoch}/{EPOCHS}",
        )
        log(f"{model_id} epoch {epoch}/{EPOCHS}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    for name, value in best_state.items():
        setattr(model, name, value)

    train_loss, train_acc = evaluate(model, train_x, train_y)
    val_loss, val_acc = evaluate(model, val_x, val_y)
    test_loss, test_acc = evaluate(model, test_x, test_y)

    metrics = {
        "train_loss": train_loss, "validation_loss": val_loss, "test_loss": test_loss,
        "train_accuracy": train_acc, "validation_accuracy": val_acc, "test_accuracy": test_acc,
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Evaluation produced non-finite metrics; candidate cannot be promoted.")
    if not should_promote(parent_validation_loss, val_loss):
        rejected_dir = STORAGE_DIR / "rejected" / model_id
        rejected_dir.mkdir(parents=True, exist_ok=True)
        rejection = {
            "id": model_id,
            "parent_id": parent["id"] if parent else None,
            "reason": "Validation loss regressed beyond the configured promotion tolerance.",
            "parent_validation_loss": parent_validation_loss,
            "promotion_max_regression": PROMOTION_MAX_REGRESSION,
            "metrics": metrics,
            "source_ids": source_ids,
            "dataset_hash": dataset_hash,
        }
        (rejected_dir / "metadata.json").write_text(json.dumps(rejection, indent=2), encoding="utf-8")
        store.update_model(
            model_id, status="rejected", completed_at=store.utcnow(),
            train_loss=train_loss, validation_loss=val_loss, test_loss=test_loss,
            train_accuracy=train_acc, validation_accuracy=val_acc,
            test_accuracy=test_acc, parameter_count=parameter_count,
            error=rejection["reason"],
        )
        store.emit("model", {"id": model_id, "status": "rejected", "metrics": metrics})
        store.set_runtime(
            status="idle", stage="waiting_for_data", current_model_id=None,
            current_epoch=0, total_epochs=0, progress=100,
            message=f"{model_id} was not promoted: validation loss did not meet the promotion gate.",
            last_candidate_id=model_id,
        )
        log(f"Rejected {model_id}: parent validation loss={parent_validation_loss:.4f}, candidate={val_loss:.4f}", "warning")
        return

    candidate_dir = MODELS_DIR / f".{model_id}.candidate"
    final_dir = MODELS_DIR / model_id
    candidate_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = candidate_dir / "model.npz"
    np.savez_compressed(
        checkpoint_path, E=model.E, W1=model.W1, b1=model.b1,
        W2=model.W2, b2=model.b2, chars=np.array(list(ALPHABET)),
        context_size=np.array(CONTEXT_SIZE),
        embedding_dim=np.array(EMBEDDING_DIM), hidden_size=np.array(HIDDEN_SIZE),
    )
    metadata = {
        "id": model_id, "parent_id": parent["id"] if parent else None,
        "best_epoch": best_epoch, "epochs": EPOCHS,
        "parameter_count": parameter_count, "dataset_hash": dataset_hash,
        "source_ids": source_ids, "source_count": len(source_ids),
        "architecture": {"context_size": CONTEXT_SIZE, "embedding_dim": EMBEDDING_DIM, "hidden_size": HIDDEN_SIZE, "vocab_size": len(ALPHABET)},
        "metrics": {"train_loss": train_loss, "validation_loss": val_loss, "test_loss": test_loss, "train_accuracy": train_acc, "validation_accuracy": val_acc, "test_accuracy": test_acc},
    }
    (candidate_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Directory rename is atomic on the same mounted filesystem. The API only
    # lists status=ready models, so incomplete candidates are never selectable.
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir.rename(final_dir)
    final_checkpoint = final_dir / "model.npz"
    store.update_model(
        model_id, status="ready", completed_at=store.utcnow(),
        train_loss=train_loss, validation_loss=val_loss, test_loss=test_loss,
        train_accuracy=train_acc, validation_accuracy=val_acc,
        test_accuracy=test_acc, parameter_count=parameter_count,
        checkpoint_path=str(final_checkpoint),
    )
    store.emit("model", {"id": model_id, "status": "ready", "metrics": metadata["metrics"]})
    store.set_runtime(
        status="idle", stage="waiting_for_data", current_model_id=None,
        current_epoch=0, total_epochs=0, progress=100,
        message=f"{model_id} published. Waiting for new licensed articles.",
        last_completed_model_id=model_id,
    )
    log(f"Published {model_id}: test_loss={test_loss:.4f}, test_accuracy={test_acc:.2f}%")


def run_forever() -> None:
    store.init_db()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # A process killed during a prior training cycle may have left a candidate
    # row behind. Mark it interrupted; published checkpoints remain untouched.
    for stale in store.list_models():
        if stale["status"] == "training":
            store.update_model(stale["id"], status="failed", completed_at=store.utcnow(), error="Trainer restarted before this candidate completed.")
    store.set_runtime(
        status="starting", stage="initializing", message="Trainer is starting",
        progress=0, current_model_id=None, current_epoch=0, total_epochs=0,
    )
    log("Continuous trainer started")
    while True:
        try:
            store.set_runtime(status="discovering", stage="collecting", message="Searching Wikipedia for new articles", progress=0)
            log("Discovering new Wikipedia articles")
            newly_inserted = 0

            def on_article(article: dict) -> None:
                nonlocal newly_inserted
                inserted_now = store.add_articles([article])
                newly_inserted += inserted_now
                if inserted_now:
                    counts_now = store.counts()
                    store.set_runtime(
                        status="discovering", stage="collecting",
                        message=f"Collected {article['title']}",
                        last_collected_title=article["title"],
                        counts=counts_now,
                    )
                    log(f"Collected {article['title']} ({article.get('topic')})")

            discovered = collect_articles(
                max_articles=15, max_chars=50_000,
                max_article_chars=5_000, max_per_topic=2,
                on_article=on_article,
            )
            # Callback stores each article as soon as it arrives so the GUI
            # can display the source inventory during discovery.
            store.add_articles(discovered)
            counts = store.counts()
            log(f"Discovery finished: {len(discovered)} fetched, {newly_inserted} new, {counts['articles']} stored")
            parent = store.latest_ready_model()
            latest_attempt = store.latest_model()
            articles = store.get_articles_with_text()
            current_train_articles = split_articles(articles)["train"] if len(articles) >= 3 else []
            current_fingerprint = dataset_fingerprint(current_train_articles) if current_train_articles else None
            dataset_changed = bool(current_fingerprint and (not parent or parent.get("dataset_hash") != current_fingerprint))
            if latest_attempt and latest_attempt.get("dataset_hash") == current_fingerprint and latest_attempt.get("status") in {"ready", "rejected"}:
                dataset_changed = False
            if counts["articles"] >= 3 and dataset_changed:
                train_candidate(articles, parent)
            else:
                store.set_runtime(
                    status="idle", stage="waiting_for_data", current_model_id=None,
                    progress=100, message="No new training-split articles found. Waiting for the next discovery cycle.",
                )
                log(f"No new training-split articles. Next discovery in {POLL_SECONDS} seconds.")
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            log("Trainer shutdown requested")
            break
        except Exception as error:
            runtime = store.get_runtime()
            current_id = runtime.get("current_model_id")
            if current_id:
                store.update_model(current_id, status="failed", completed_at=store.utcnow(), error=str(error))
                store.emit("model", {"id": current_id, "status": "failed", "error": str(error)})
            log(f"Training cycle failed: {type(error).__name__}: {error}", "error")
            store.set_runtime(status="error", stage="error", current_model_id=None, message=str(error), last_error=str(error))
            time.sleep(min(POLL_SECONDS, 300))


if __name__ == "__main__":
    run_forever()
