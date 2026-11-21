# The tokenizer is responsible for converting text into numbers that our neural network can process.

class CharacterTokenizer:
    def __init__(self, text: str):
        # Build a vocabulary of unique characters.
        self.chars = sorted(set(text))

        # Create character-to-ID mapping.
        self.char_to_id = {
            char: idx for idx, char in enumerate(self.chars)
        }

        # Create ID-to-character mapping.
        self.id_to_char = {
            idx: char for idx, char in enumerate(self.chars)
        }

        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Convert text into a list of integer IDs."""
        return [self.char_to_id[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        """Convert integer IDs back into text."""
        return "".join(self.id_to_char[token] for token in tokens)
