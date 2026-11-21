
from src.tokenizer import CharacterTokenizer
from src.dataset import create_dataset


def test_dataset():
    text = "hello"

    tokenizer = CharacterTokenizer(text)
    tokens = tokenizer.encode(text)

    x, y = create_dataset(tokens, context_size=2)

    assert x.shape == (3, 2)
    assert y.shape == (3,)

    assert list(x[0]) == tokens[:2]
    assert y[0] == tokens[2]
