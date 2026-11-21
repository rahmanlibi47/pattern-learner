import numpy as np

from src.model import PatternLearner


def load_model(path="artifacts/pattern_learner.npz"):
    checkpoint = np.load(path, allow_pickle=False)

    chars = checkpoint["chars"].tolist()
    context_size = int(checkpoint["context_size"])
    embedding_dim = int(checkpoint["embedding_dim"])
    hidden_size = int(checkpoint["hidden_size"])

    model = PatternLearner(
        vocab_size=len(chars),
        context_size=context_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
    )

    model.E = checkpoint["E"]
    model.W1 = checkpoint["W1"]
    model.b1 = checkpoint["b1"]
    model.W2 = checkpoint["W2"]
    model.b2 = checkpoint["b2"]

    char_to_id = {char: idx for idx, char in enumerate(chars)}

    id_to_char = {idx: char for idx, char in enumerate(chars)}

    return model, char_to_id, id_to_char, context_size


def generate(model, char_to_id, id_to_char, context_size, prompt, max_new_tokens=30):

    if len(prompt) < context_size:
        raise ValueError(f"Prompt must contain at least {context_size} characters")

    unknown = set(prompt) - set(char_to_id)

    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")

    result = prompt

    for step in range(max_new_tokens):
        context = result[-context_size:]

        x = np.array([[char_to_id[char] for char in context]])

        logits = model.forward(x)

        shifted = logits[0] - np.max(logits[0])
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()

        predicted_id = sample_next_character(
            logits[0],
            temperature=0.8,
        )
        next_char = id_to_char[predicted_id]

        top_indices = np.argsort(probabilities)[::-1][:3]

        predictions = [
            f"{repr(id_to_char[int(i)])}: {probabilities[i] * 100:.1f}%"
            for i in top_indices
        ]

        print(
            f"Step {step + 1:02d} | "
            f"Context: {repr(context)} | "
            f"Predicted: {repr(next_char)} | "
            f"Top 3: {', '.join(predictions)}"
        )

        result += next_char

    return result


def sample_next_character(logits, temperature=0.8):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    scaled_logits = logits / temperature

    shifted = scaled_logits - np.max(scaled_logits)

    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()

    predicted_id = np.random.choice(
        len(probabilities),
        p=probabilities,
    )

    return int(predicted_id)


def inspect_predictions(model, char_to_id, id_to_char, context_size, prompt):
    context = prompt[-context_size:]

    x = np.array([[char_to_id[char] for char in context]])

    logits = model.forward(x)

    # Convert logits to probabilities.
    shifted = logits[0] - np.max(logits[0])
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()

    # Display the five most probable characters.
    top_indices = np.argsort(probabilities)[::-1][:5]

    print(f"\nContext: {repr(context)}")
    print("Next character predictions:")

    for idx in top_indices:
        char = id_to_char[int(idx)]
        probability = probabilities[idx] * 100

        print(f"  {repr(char):6} {probability:6.2f}%")


if __name__ == "__main__":
    model, char_to_id, id_to_char, context_size = load_model()

    print("PATTERN LEARNER")
    print("=" * 40)
    print("Context size:", context_size)
    print("Vocabulary size:", len(char_to_id))

    prompt = input("\nEnter a starting prompt: ").lower()

    inspect_predictions(
        model,
        char_to_id,
        id_to_char,
        context_size,
        prompt,
    )

    output = generate(
        model,
        char_to_id,
        id_to_char,
        context_size,
        prompt,
        max_new_tokens=100,
    )

    print("\nGenerated text:")
    print(output)
