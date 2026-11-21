from pathlib import Path

from src.database import store
from src.training import worker


def test_split_articles_produces_three_nonempty_splits():
    articles = [{"id": f"article-{i}", "text": "the quick brown fox " * 40} for i in range(11)]
    splits = worker.split_articles(articles)
    assert set(splits) == {"train", "validation", "test"}
    assert all(splits.values())
    assert sum(len(value) for value in splits.values()) == 11


def test_incremental_candidate_is_published_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.sqlite3")
    monkeypatch.setattr(worker, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(worker, "EPOCHS", 1)
    monkeypatch.setattr(worker, "BATCH_SIZE", 64)
    store.init_db()

    texts = [
        ("the quick brown fox jumps over the lazy dog. " * 35),
        ("science studies nature and technology. " * 40),
        ("mathematics uses numbers and patterns. " * 40),
    ]
    articles = [
        {
            "title": f"Synthetic test article {i}",
            "text": text,
            "url": f"https://example.test/{i}",
            "license": "Test license",
            "license_url": "https://example.test/license",
            "topic": "test",
        }
        for i, text in enumerate(texts)
    ]
    assert store.add_articles(articles) == 3

    worker.train_candidate(store.get_articles_with_text(), parent=None)

    models = store.list_models()
    assert len(models) == 1
    assert models[0]["status"] == "ready"
    assert models[0]["parameter_count"] == 521
    assert Path(models[0]["checkpoint_path"]).is_file()
    assert models[0]["test_loss"] is not None


def test_article_splits_are_persisted_and_never_reassigned(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "split.sqlite3")
    store.init_db()
    articles = [
        {
            "title": f"Source {i}",
            "text": ("the quick brown fox and the clever dog. " * 20) + str(i),
            "url": f"https://example.test/source-{i}",
            "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "topic": "test",
        }
        for i in range(5)
    ]
    store.add_articles(articles)
    first = {item["id"]: item["split"] for item in store.list_articles()}
    assert set(first.values()) == {"train", "validation", "test"}

    extra = {
        "title": "Source extra", "text": "the science and nature. " * 40,
        "url": "https://example.test/extra", "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/", "topic": "test",
    }
    store.add_articles([extra])
    second = {item["id"]: item["split"] for item in store.list_articles()}
    assert all(second[article_id] == split for article_id, split in first.items())


def test_validation_gate_rejects_regressions(monkeypatch):
    monkeypatch.setattr(worker, "PROMOTION_MAX_REGRESSION", 0.02)
    assert worker.should_promote(None, 2.5)
    assert worker.should_promote(2.5, 2.51)
    assert not worker.should_promote(2.5, 2.53)
    assert not worker.should_promote(2.5, float("nan"))
