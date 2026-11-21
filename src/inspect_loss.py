
from pathlib import Path

from src.tokenizer import CharacterTokenizer
from src.dataset import create_dataset
from src.model import PatternLearner
from src.loss import softmax, cross_entropy_loss


text = Path("data/training.txt").read_text(encoding="utf-8")

tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode(text)

x, y = create_dataset(tokens, context_size=4)

model = PatternLearner(
    vocab_size=tokenizer.vocab_size
)

logits = model.forward(x)

probabilities = softmax(logits)

loss = cross_entropy_loss(logits, y)

print("Initial loss:", loss)

print("\nFirst prediction:")

input_text = tokenizer.decode(x[0].tolist())
expected = tokenizer.decode([int(y[0])])

predicted_id = probabilities[0].argmax()
predicted_char = tokenizer.decode([int(predicted_id)])

print("Input:", input_text)
print("Expected:", repr(expected))
print("Predicted:", repr(predicted_char))
print("Probability:", probabilities[0][predicted_id])

print("\nProbability sum:", probabilities[0].sum())
