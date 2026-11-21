
import numpy as np

from pathlib import Path
from src.tokenizer import CharacterTokenizer
from src.dataset import create_dataset
from src.model import PatternLearner


text = Path("data/training.txt").read_text(encoding="utf-8")

tokenizer = CharacterTokenizer(text)
tokens = tokenizer.encode(text)

x, y = create_dataset(tokens, context_size=4)

model = PatternLearner(
    vocab_size=tokenizer.vocab_size
)

predictions = model.forward(x)

print("Training examples:", len(x))
print("Input shape:", x.shape)
print("Output shape:", predictions.shape)
print("Total parameters:", model.count_parameters())

print("\nFirst prediction:")
print("Input:", tokenizer.decode(x[0].tolist()))
print("Expected:", tokenizer.decode([int(y[0])]))
print("Predicted character ID:", np.argmax(predictions[0]))

print("\nParameter breakdown:")
for name, count in model.parameter_breakdown().items():
    print(f"{name}: {count}")
