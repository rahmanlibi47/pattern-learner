
from src.tokenizer import CharacterTokenizer


def test_encode_decode():
    text = "hello world. hello pattern learner."

    tokenizer = CharacterTokenizer(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


def test_vocabulary():
    text = "hello"

    tokenizer = CharacterTokenizer(text)

    assert tokenizer.vocab_size == len(set(text))


def test_unknown_character():
    tokenizer = CharacterTokenizer("hello")

    try:
        tokenizer.encode("hello!")
        assert False, "Expected unknown character to raise KeyError"
    except KeyError:
        pass
