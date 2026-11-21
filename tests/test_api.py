import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app
from src.database import store
from src.model import PatternLearner


def test_ready_checkpoint_can_generate_without_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "api.sqlite3")
    store.init_db()

    chars = list("abcdefghijklmnopqrstuvwxyz .,!?;:'-")
    model = PatternLearner(len(chars), context_size=8, embedding_dim=2, hidden_size=8)
    checkpoint = tmp_path / "model.npz"
    np.savez_compressed(
        checkpoint,
        E=model.E, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2,
        chars=np.array(chars), context_size=np.array(8),
        embedding_dim=np.array(2), hidden_size=np.array(8),
    )
    store.create_model("model-test", None, 1, [], "test-hash")
    store.update_model("model-test", status="ready", checkpoint_path=str(checkpoint), parameter_count=model.count_parameters())
    store.create_model("model-candidate", "model-test", 5, [], "candidate-hash")

    with TestClient(app) as client:
        response = client.post("/api/generate", json={
            "model_id": "model-test",
            "prompt": "the test",
            "max_new_tokens": 8,
            "temperature": 0,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["model_id"] == "model-test"
        assert body["generated_text"].startswith("the test")
        assert len(body["generated_text"]) == 16

        candidate_response = client.post("/api/generate", json={
            "model_id": "model-candidate",
            "prompt": "the test",
            "max_new_tokens": 8,
            "temperature": 0,
        })
        assert candidate_response.status_code == 404
