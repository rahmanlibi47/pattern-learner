
from pathlib import Path
import numpy as np

from src.tokenizer import CharacterTokenizer
from src.dataset import create_dataset
from src.model import PatternLearner
from src.loss import softmax, cross_entropy_loss


text = Path("data/training.txt").read_text(encoding="utf-8")

tokenizer = CharacterTokenizer(text)
tokens = tokenizer.encode(text)

x, y = create_dataset(tokens, context_size=4)

model = PatternLearner(
    vocab_size=tokenizer.vocab_size
)

# Forward pass
logits = model.forward(x)

# Calculate probabilities
probabilities = softmax(logits)

# Calculate loss
loss = cross_entropy_loss(logits, y)

# Calculate gradient of loss with respect to logits
dlogits = probabilities.copy()

dlogits[np.arange(len(y)), y] -= 1

dlogits /= len(y)

# Backward pass
gradients = model.backward(x, dlogits)

print("Loss:", loss)

print("\nGradient shapes:")

for name, gradient in gradients.items():
    print(f"{name}: {gradient.shape}")

print("\nGradient magnitudes:")

for name, gradient in gradients.items():
    print(
        f"{name}: "
        f"mean={np.mean(np.abs(gradient)):.8f}, "
        f"max={np.max(np.abs(gradient)):.8f}"
    )

print("\nAll gradients finite:",
      all(np.all(np.isfinite(g))
          for g in gradients.values()))
