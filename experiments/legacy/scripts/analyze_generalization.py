
from pathlib import Path
from collections import defaultdict
import numpy as np

from src.generate import load_model
from src.dataset import create_dataset


model, char_to_id, id_to_char, context_size = load_model()

train_text = Path("data/training.txt").read_text(
    encoding="utf-8"
).strip()

test_text = Path("data/test.txt").read_text(
    encoding="utf-8"
).strip()


def encode(text):
    unknown = set(text) - set(char_to_id)

    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")

    return [char_to_id[c] for c in text]


train_tokens = encode(train_text)
test_tokens = encode(test_text)

train_x, train_y = create_dataset(train_tokens, context_size)
test_x, test_y = create_dataset(test_tokens, context_size)

# Collect training contexts and their observed next characters.
known_transitions = defaultdict(set)

for context, target in zip(train_x, train_y):
    key = tuple(context.tolist())
    known_transitions[key].add(int(target))

# Divide test examples into three categories.
categories = {
    "seen": [],
    "unseen": [],
    "conflicting": [],
}

for context, target in zip(test_x, test_y):
    key = tuple(context.tolist())
    target = int(target)

    if key not in known_transitions:
        categories["unseen"].append((context, target))
    elif target not in known_transitions[key]:
        categories["conflicting"].append((context, target))
    else:
        categories["seen"].append((context, target))


def evaluate(examples, name):
    print(f"\n{name}")
    print("-" * 40)

    if not examples:
        print("No examples found.")
        return

    x = np.array([item[0] for item in examples])
    y = np.array([item[1] for item in examples])

    logits = model.forward(x)
    predictions = np.argmax(logits, axis=1)

    accuracy = np.mean(predictions == y) * 100

    print("Examples:", len(y))
    print(f"Accuracy: {accuracy:.2f}%")

    for i in range(min(5, len(y))):
        context = "".join(
            id_to_char[int(c)] for c in x[i]
        )

        expected = id_to_char[int(y[i])]
        predicted = id_to_char[int(predictions[i])]

        print(
            f"{context!r} -> "
            f"Expected: {expected!r}, "
            f"Predicted: {predicted!r}"
        )


print("GENERALIZATION ANALYSIS")
print("=" * 40)

print("Training examples:", len(train_x))
print("Test examples:", len(test_x))

for name, examples in categories.items():
    print(f"{name.capitalize()} transitions: {len(examples)}")

evaluate(categories["seen"], "Seen transitions")
evaluate(categories["unseen"], "Unseen contexts")
evaluate(categories["conflicting"], "Conflicting transitions")
