
import numpy as np


class PatternLearner:
    def __init__(
        self,
        vocab_size: int,
        context_size: int = 4,
        embedding_dim: int = 8,
        hidden_size: int = 18,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        input_size = context_size * embedding_dim

        # Character embedding table
        self.E = rng.normal(
            0, 0.1, (vocab_size, embedding_dim)
        )

        # Hidden layer
        self.W1 = rng.normal(
            0, np.sqrt(2 / input_size),
            (input_size, hidden_size)
        )
        self.b1 = np.zeros(hidden_size)

        # Output layer
        self.W2 = rng.normal(
            0, np.sqrt(2 / hidden_size),
            (hidden_size, vocab_size)
        )
        self.b2 = np.zeros(vocab_size)

    def count_parameters(self):
        return sum(
            p.size for p in
            [self.E, self.W1, self.b1, self.W2, self.b2]
        )

    
    def forward(self, x):
        embeddings = self.E[x]

        flattened = embeddings.reshape(x.shape[0], -1)

        z1 = flattened @ self.W1 + self.b1
        hidden = np.tanh(z1)

        logits = hidden @ self.W2 + self.b2

        # Save intermediate values for backpropagation
        self._cache = {
            "x": x,
            "embeddings": embeddings,
            "flattened": flattened,
            "hidden": hidden,
        }

        return logits

    def backward(self, x, dlogits):
        cache = self._cache

        hidden = cache["hidden"]
        flattened = cache["flattened"]

        # Output layer gradients
        dW2 = hidden.T @ dlogits
        db2 = np.sum(dlogits, axis=0)

        # Backpropagate into hidden layer
        dhidden = dlogits @ self.W2.T

        dz1 = dhidden * (1 - hidden ** 2)

        # Hidden layer gradients
        dW1 = flattened.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Backpropagate into embeddings
        dflattened = dz1 @ self.W1.T

        dembeddings = dflattened.reshape(
            x.shape[0],
            self.context_size,
            self.embedding_dim,
        )

        dE = np.zeros_like(self.E)

        # Accumulate gradients for repeated characters
        np.add.at(
            dE,
            x,
            dembeddings,
        )

        return {
            "E": dE,
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

    def parameter_breakdown(self):
        return {
            "embeddings": self.E.size,
            "hidden_weights": self.W1.size,
            "hidden_biases": self.b1.size,
            "output_weights": self.W2.size,
            "output_biases": self.b2.size,
            "total": self.count_parameters(),
        }