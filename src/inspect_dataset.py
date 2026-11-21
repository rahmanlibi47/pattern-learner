
from pathlib import Path

from src.tokenizer import CharacterTokenizer
from src.dataset import create_dataset

text = Path("data/training.txt").read_text(encoding="utf-8")

tokenizer = CharacterTokenizer(text)
tokens = tokenizer.encode(text)

context_size = 4

x, y = create_dataset(tokens, context_size)

print("Total examples:", len(x))
print()

for i in range(min(10, len(x))):
    input_text = tokenizer.decode(x[i].tolist())
    target_text = tokenizer.decode([int(y[i])])

    print(f"Input: {input_text!r} -> Target: {target_text!r}")
