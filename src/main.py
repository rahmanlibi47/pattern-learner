
from pathlib import Path
from src.tokenizer import CharacterTokenizer

text = Path("data/training.txt").read_text(encoding="utf-8")

tokenizer = CharacterTokenizer(text)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("Original text:", text)
print("Vocabulary:", tokenizer.chars)
print("Vocabulary size:", tokenizer.vocab_size)
print("Encoded:", encoded)
print("Decoded:", decoded)
