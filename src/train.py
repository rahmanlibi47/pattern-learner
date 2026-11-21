
from pathlib import Path
import numpy as np

from src.optimizer import Adam
from src.tokenizer import CharacterTokenizer
from src.dataset import create_dataset
from src.model import PatternLearner
from src.loss import softmax, cross_entropy_loss


# Configuration
LEARNING_RATE = 0.01
EPOCHS = 10000
CONTEXT_SIZE = 8
EMBEDDING_DIM = 4
HIDDEN_SIZE = 13

# Load training data
text = Path("data/training.txt").read_text(encoding="utf-8")

tokenizer = CharacterTokenizer(text)
tokens = tokenizer.encode(text)

x, y = create_dataset(tokens, CONTEXT_SIZE)

# Initialize model
model = PatternLearner(
    vocab_size=tokenizer.vocab_size,
    context_size=CONTEXT_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
)

print("Pattern Learner Training")
print("-" * 40)
print("Training examples:", len(x))
print("Parameters:", model.count_parameters())

# Initial Adam Optimizer
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




# Initial loss
initial_loss = cross_entropy_loss(model.forward(x), y)

print(f"Initial loss: {initial_loss:.6f}")

# Training loop
for epoch in range(1, EPOCHS + 1):

    # Forward propagation
    logits = model.forward(x)

    # Calculate probabilities
    probabilities = softmax(logits)

    # Calculate loss
    loss = cross_entropy_loss(logits, y)

    # Calculate gradients
    dlogits = probabilities.copy()

    dlogits[np.arange(len(y)), y] -= 1
    dlogits /= len(y)

    gradients = model.backward(x, dlogits)

    # Gradient descent
    optimizer.step(parameters, gradients)

    # Report progress
    if epoch == 1 or epoch % 500 == 0:
        print(
            f"Epoch {epoch:4d}/{EPOCHS} "
            f"| Loss: {loss:.6f}"
        )

# Final evaluation
final_logits = model.forward(x)
final_loss = cross_entropy_loss(final_logits, y)

predictions = np.argmax(final_logits, axis=1)
accuracy = np.mean(predictions == y) * 100

print("\nTraining completed!")
print(f"Initial loss: {initial_loss:.6f}")
print(f"Final loss:   {final_loss:.6f}")
print(f"Accuracy:     {accuracy:.2f}%")


# Save trained parameters
checkpoint_dir = Path("artifacts")
checkpoint_dir.mkdir(exist_ok=True)

np.savez(
    checkpoint_dir / "pattern_learner.npz",
    E=model.E,
    W1=model.W1,
    b1=model.b1,
    W2=model.W2,
    b2=model.b2,
    chars=np.array(tokenizer.chars),
    context_size=np.array(CONTEXT_SIZE),
    embedding_dim=np.array(model.embedding_dim),
    hidden_size=np.array(model.hidden_size),
)

print("\nModel saved successfully!")