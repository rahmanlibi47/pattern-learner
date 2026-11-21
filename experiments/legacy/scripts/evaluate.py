
from pathlib import Path
import numpy as np

from src.generate import load_model
from src.dataset import create_dataset
from src.loss import cross_entropy_loss


# Load saved model
model, char_to_id, id_to_char, context_size = load_model()


def encode_text(text):
    unknown = set(text) - set(char_to_id)

    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")

    return [char_to_id[c] for c in text]


def evaluate(text, name):
    tokens = encode_text(text)

    x, y = create_dataset(tokens, context_size)

    if len(x) == 0:
        raise ValueError("Not enough text for evaluation")

    logits = model.forward(x)
    probabilities = np.exp(
        logits - np.max(logits, axis=1, keepdims=True)
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    loss = cross_entropy_loss(logits, y)

    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions == y)

    print(f"\n{name}")
    print("-" * 40)
    print("Examples:", len(x))
    print(f"Loss: {loss:.6f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nSample predictions:")

    for i in range(min(10, len(x))):
        context = "".join(
            id_to_char[int(token)] for token in x[i]
        )

        expected = id_to_char[int(y[i])]
        predicted = id_to_char[int(predictions[i])]

        status = "PASS" if expected == predicted else "FAIL"

        print(
            f"{context!r} -> "
            f"Expected: {expected!r}, "
            f"Predicted: {predicted!r} [{status}]"
        )

    return {
        "loss": loss,
        "accuracy": accuracy,
        "examples": len(x),
    }


# Load original training data
training_text = Path("data/training.txt").read_text(
    encoding="utf-8"
)

# Load completely separate test data
test_text = Path("data/test.txt").read_text(
    encoding="utf-8"
).strip()

print("PATTERN LEARNER EVALUATION")
print("=" * 40)
print("Parameters:", model.count_parameters())
print("Context size:", context_size)
print("Vocabulary size:", len(char_to_id))

train_results = evaluate(training_text, "TRAINING RESULTS")
test_results = evaluate(test_text, "TEST RESULTS")

print("\nFINAL COMPARISON")
print("=" * 40)

print(
    f"Training accuracy: "
    f"{train_results['accuracy'] * 100:.2f}%"
)

print(
    f"Test accuracy: "
    f"{test_results['accuracy'] * 100:.2f}%"
)

print(
    f"Training loss: {train_results['loss']:.6f}"
)

print(
    f"Test loss: {test_results['loss']:.6f}"
)
