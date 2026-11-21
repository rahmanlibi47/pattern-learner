
import numpy as np


def softmax(logits):
    # Numerical stability
    shifted = logits - np.max(
        logits, axis=1, keepdims=True
    )

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )


def cross_entropy_loss(logits, targets):
    probabilities = softmax(logits)

    batch_size = len(targets)

    correct_probabilities = probabilities[
        np.arange(batch_size), targets
    ]

    loss = -np.mean(
        np.log(correct_probabilities + 1e-12)
    )

    return loss
