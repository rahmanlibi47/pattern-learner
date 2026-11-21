
import numpy as np

from src.model import PatternLearner


def test_model():
    model = PatternLearner(vocab_size=14)

    x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ])

    output = model.forward(x)

    assert output.shape == (2, 14)
    assert np.all(np.isfinite(output))


def test_parameter_count():
    model = PatternLearner(vocab_size=14)

    assert model.count_parameters() == 972
