from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.model import PatternLearner


@lru_cache(maxsize=8)
def load_model(model_id: str, checkpoint_path: str) -> tuple[PatternLearner, dict[str, int], dict[int, str], int]:
    """Load and cache an immutable, published checkpoint by model version."""
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        chars = checkpoint["chars"].tolist()
        context_size = int(checkpoint["context_size"])
        embedding_dim = int(checkpoint["embedding_dim"])
        hidden_size = int(checkpoint["hidden_size"])
        model = PatternLearner(
            vocab_size=len(chars), context_size=context_size,
            embedding_dim=embedding_dim, hidden_size=hidden_size,
        )
        model.E = checkpoint["E"].copy()
        model.W1 = checkpoint["W1"].copy()
        model.b1 = checkpoint["b1"].copy()
        model.W2 = checkpoint["W2"].copy()
        model.b2 = checkpoint["b2"].copy()
    return model, {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}, context_size


def generate_text(
    model_id: str,
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    seed: int | None = None,
) -> str:
    model, char_to_id, id_to_char, context_size = load_model(model_id, checkpoint_path)
    prompt = prompt.lower()
    if len(prompt) < context_size:
        raise ValueError(f"Prompt must contain at least {context_size} characters.")
    unknown = sorted(set(prompt) - set(char_to_id))
    if unknown:
        raise ValueError(f"Unsupported characters: {unknown}")

    rng = np.random.default_rng(seed)
    result = prompt
    for _ in range(max_new_tokens):
        context = result[-context_size:]
        x = np.array([[char_to_id[char] for char in context]], dtype=np.int64)
        logits = model.forward(x)[0]
        if temperature == 0:
            next_id = int(np.argmax(logits))
        else:
            scaled = logits / temperature
            scaled -= np.max(scaled)
            probabilities = np.exp(scaled)
            probabilities /= probabilities.sum()
            next_id = int(rng.choice(len(probabilities), p=probabilities))
        result += id_to_char[next_id]
    return result
