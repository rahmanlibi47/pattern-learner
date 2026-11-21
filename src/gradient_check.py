
from pathlib import Path
import numpy as np

from src.tokenizer import CharacterTokenizer
from src.dataset import create_dataset
from src.model import PatternLearner
from src.loss import cross_entropy_loss


text = Path("data/training.txt").read_text(encoding="utf-8")

tokenizer = CharacterTokenizer(text)
tokens = tokenizer.encode(text)

x, y = create_dataset(tokens, context_size=4)

model = PatternLearner(vocab_size=tokenizer.vocab_size)

# Analytical gradients
logits = model.forward(x)
probabilities = np.exp(
    logits - np.max(logits, axis=1, keepdims=True)
)
probabilities /= probabilities.sum(axis=1, keepdims=True)

dlogits = probabilities.copy()
dlogits[np.arange(len(y)), y] -= 1
dlogits /= len(y)

gradients = model.backward(x, dlogits)

# Numerical gradient checking
parameters = {
    "E": model.E,
    "W1": model.W1,
    "b1": model.b1,
    "W2": model.W2,
    "b2": model.b2,
}

epsilon = 1e-5

print("NUMERICAL GRADIENT CHECK")
print("-" * 55)

for name, parameter in parameters.items():
    analytical = gradients[name]
    numerical = np.zeros_like(parameter)

    # Check up to five randomly selected parameters
    rng = np.random.default_rng(42)
    indices = list(zip(*np.where(np.ones(parameter.shape, dtype=bool))))
    selected = rng.choice(
        len(indices),
        size=min(5, len(indices)),
        replace=False,
    )

    for index_number in selected:
        index = indices[index_number]

        original = parameter[index]

        parameter[index] = original + epsilon
        loss_plus = cross_entropy_loss(model.forward(x), y)

        parameter[index] = original - epsilon
        loss_minus = cross_entropy_loss(model.forward(x), y)

        parameter[index] = original

        numerical[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        # Compare this individual gradient
        analytic_value = analytical[index]
        numerical_value = numerical[index]

        relative_error = abs(
            analytic_value - numerical_value
        ) / max(
            1e-8,
            abs(analytic_value) + abs(numerical_value)
        )

        print(
            f"{name}{index}: "
            f"analytical={analytic_value:.8f}, "
            f"numerical={numerical_value:.8f}, "
            f"error={relative_error:.8e}"
        )

print("\nGradient checking completed.")
