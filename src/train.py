
import json
from pathlib import Path

import numpy as np

from src.optimizer import Adam
from src.tokenizer import CharacterTokenizer
from src.dataset import create_dataset
from src.model import PatternLearner
from src.loss import softmax, cross_entropy_loss

from src.data.collector import collect_articles
from src.data.prepare import prepare_articles


# ============================================================
# Configuration
# ============================================================

LEARNING_RATE = 0.003
EPOCHS = 150
BATCH_SIZE = 128

CONTEXT_SIZE = 8
EMBEDDING_DIM = 2
HIDDEN_SIZE = 8

SEED = 42
MAX_PARAMETERS = 1000

# Must match the characters retained by normalize_text().
CHARACTER_VOCAB = "abcdefghijklmnopqrstuvwxyz .,!?;:'-"


# ============================================================
# Dataset preparation
# ============================================================

def build_dataset(articles, tokenizer, context_size):
    """Build character sequences without crossing article boundaries."""

    inputs = []
    targets = []

    for article in articles:
        tokens = tokenizer.encode(article["text"])

        x, y = create_dataset(tokens, context_size)

        if len(x) == 0:
            continue

        inputs.append(x)
        targets.append(y)

    if not inputs:
        raise ValueError("No training examples were generated.")

    return (
        np.concatenate(inputs, axis=0),
        np.concatenate(targets, axis=0),
    )


def evaluate(model, x, y, batch_size=512):
    """Calculate average cross-entropy loss and accuracy."""

    total_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, len(x), batch_size):
        batch_x = x[start:start + batch_size]
        batch_y = y[start:start + batch_size]

        logits = model.forward(batch_x)

        loss = cross_entropy_loss(logits, batch_y)

        predictions = np.argmax(logits, axis=1)

        total_loss += loss * len(batch_y)
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)

    return total_loss / total, correct / total * 100


# ============================================================
# Collect online data
# ============================================================

print("\nPATTERN LEARNER")
print("=" * 50)

print("\nCollecting online training data...")

articles = collect_articles(
    max_articles=15,
    max_chars=50_000,
    max_article_chars=5_000,
    max_per_topic=2,
)

if len(articles) < 3:
    raise RuntimeError(
        "Not enough articles for training, validation and testing."
    )

datasets = prepare_articles(articles)

train_articles = datasets["train"]
validation_articles = datasets["validation"]
test_articles = datasets["test"]


# ============================================================
# Tokenizer
# ============================================================

# Use a fixed alphabet matching our normalization rules.
# This prevents unseen validation/test characters from
# causing tokenizer errors.

tokenizer = CharacterTokenizer(CHARACTER_VOCAB)

print("\nDataset preparation")
print("-" * 50)

for name, split in datasets.items():
    characters = sum(
        len(article["text"]) for article in split
    )

    print(
        f"{name.capitalize():12} "
        f"| Articles: {len(split):2} "
        f"| Characters: {characters:6}"
    )

print("\nVocabulary size:", tokenizer.vocab_size)


# ============================================================
# Create character-level datasets
# ============================================================

train_x, train_y = build_dataset(
    train_articles, tokenizer, CONTEXT_SIZE
)

validation_x, validation_y = build_dataset(
    validation_articles, tokenizer, CONTEXT_SIZE
)

test_x, test_y = build_dataset(
    test_articles, tokenizer, CONTEXT_SIZE
)

print("\nTraining examples:", len(train_x))
print("Validation examples:", len(validation_x))
print("Test examples:", len(test_x))


# ============================================================
# Initialize model
# ============================================================

model = PatternLearner(
    vocab_size=tokenizer.vocab_size,
    context_size=CONTEXT_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    seed=SEED,
)

parameter_count = model.count_parameters()

print("\nModel architecture")
print("-" * 50)

print("Context size:", CONTEXT_SIZE)
print("Embedding dimension:", EMBEDDING_DIM)
print("Hidden size:", HIDDEN_SIZE)
print("Parameters:", parameter_count)

if parameter_count > MAX_PARAMETERS:
    raise ValueError(
        f"Model has {parameter_count} parameters. "
        f"Maximum allowed: {MAX_PARAMETERS}"
    )


# ============================================================
# Initialize optimizer
# ============================================================

optimizer = Adam(
    learning_rate=LEARNING_RATE
)

parameters = {
    "E": model.E,
    "W1": model.W1,
    "b1": model.b1,
    "W2": model.W2,
    "b2": model.b2,
}

rng = np.random.default_rng(SEED)


# ============================================================
# Initial evaluation
# ============================================================

initial_train_loss, initial_train_accuracy = evaluate(
    model, train_x, train_y
)

initial_validation_loss, _ = evaluate(
    model, validation_x, validation_y
)

print("\nInitial training loss:", round(initial_train_loss, 6))
print("Initial validation loss:", round(initial_validation_loss, 6))


# ============================================================
# Training loop
# ============================================================

best_validation_loss = float("inf")
best_epoch = 0
best_state = None

for epoch in range(1, EPOCHS + 1):

    # Shuffle training examples.
    indices = rng.permutation(len(train_x))

    for start in range(0, len(train_x), BATCH_SIZE):

        batch_indices = indices[start:start + BATCH_SIZE]

        batch_x = train_x[batch_indices]
        batch_y = train_y[batch_indices]

        # Forward propagation.
        logits = model.forward(batch_x)

        # Convert logits into probabilities.
        probabilities = softmax(logits)

        # Calculate gradients of cross-entropy loss.
        dlogits = probabilities.copy()

        dlogits[np.arange(len(batch_y)), batch_y] -= 1
        dlogits /= len(batch_y)

        gradients = model.backward(batch_x, dlogits)

        # Update parameters using Adam.
        optimizer.step(parameters, gradients)

    # Evaluate after each epoch.
    train_loss, train_accuracy = evaluate(
        model, train_x, train_y
    )

    validation_loss, validation_accuracy = evaluate(
        model, validation_x, validation_y
    )

    print(
        f"Epoch {epoch:3d}/{EPOCHS} "
        f"| Train loss: {train_loss:.4f} "
        f"| Train acc: {train_accuracy:.2f}% "
        f"| Val loss: {validation_loss:.4f} "
        f"| Val acc: {validation_accuracy:.2f}%"
    )

    # Preserve the model with the best validation loss.
    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss
        best_epoch = epoch

        best_state = {
            name: parameter.copy()
            for name, parameter in parameters.items()
        }


# ============================================================
# Restore best model
# ============================================================

for name, value in best_state.items():
    setattr(model, name, value)


# ============================================================
# Final evaluation
# ============================================================

train_loss, train_accuracy = evaluate(
    model, train_x, train_y
)

validation_loss, validation_accuracy = evaluate(
    model, validation_x, validation_y
)

test_loss, test_accuracy = evaluate(
    model, test_x, test_y
)

print("\nFINAL EVALUATION")
print("=" * 50)

print("Best epoch:", best_epoch)

print(f"Training loss:   {train_loss:.6f}")
print(f"Training acc:    {train_accuracy:.2f}%")

print(f"Validation loss: {validation_loss:.6f}")
print(f"Validation acc:  {validation_accuracy:.2f}%")

print(f"Test loss:       {test_loss:.6f}")
print(f"Test acc:        {test_accuracy:.2f}%")


# ============================================================
# Save checkpoint and source attribution
# ============================================================

checkpoint_dir = Path("artifacts")
checkpoint_dir.mkdir(exist_ok=True)

source_metadata = {
    name: [
        {
            "title": article["title"],
            "url": article["url"],
            "license": article["license"],
            "license_url": article["license_url"],
            "topic": article["topic"],
        }
        for article in split
    ]
    for name, split in datasets.items()
}

np.savez(
    checkpoint_dir / "pattern_learner.npz",

    E=model.E,
    W1=model.W1,
    b1=model.b1,
    W2=model.W2,
    b2=model.b2,

    chars=np.array(tokenizer.chars),

    context_size=np.array(CONTEXT_SIZE),
    embedding_dim=np.array(EMBEDDING_DIM),
    hidden_size=np.array(HIDDEN_SIZE),

    sources=np.array(json.dumps(source_metadata)),

    best_epoch=np.array(best_epoch),
    train_loss=np.array(train_loss),
    validation_loss=np.array(validation_loss),
    test_loss=np.array(test_loss),
)

print("\nModel saved successfully!")
print("Checkpoint: artifacts/pattern_learner.npz")
