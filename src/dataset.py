
import numpy as np


def create_dataset(tokens: list[int], context_size: int):
    inputs = []
    targets = []

    for i in range(len(tokens) - context_size):
        x = tokens[i : i + context_size]
        y = tokens[i + context_size]

        inputs.append(x)
        targets.append(y)

    return np.array(inputs), np.array(targets)
