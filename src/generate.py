
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

    char_to_id = {
        char: idx for idx, char in enumerate(chars)
    }

    id_to_char = {
        idx: char for idx, char in enumerate(chars)
    }

    return model, char_to_id, id_to_char, context_size



def generate(model, char_to_id, id_to_char,
             context_size, prompt, max_new_tokens=30):

    if len(prompt) < context_size:
        raise ValueError(
            f"Prompt must contain at least {context_size} characters"
        )

    unknown = set(prompt) - set(char_to_id)

    if unknown:
        raise ValueError(f"Unknown characters: {unknown}")

    result = prompt

    for _ in range(max_new_tokens):

        context = result[-context_size:]

        x = np.array([
            [char_to_id[char] for char in context]
        ])

        logits = model.forward(x)

        predicted_id = int(np.argmax(logits[0]))

        next_char = id_to_char[predicted_id]

        result += next_char

    return result


if __name__ == "__main__":
    model, char_to_id, id_to_char, context_size = load_model()

    prompt = "hell"

    output = generate(
        model,
        char_to_id,
        id_to_char,
        context_size,
        prompt,
        max_new_tokens=30,
    )

    print("Prompt:", repr(prompt))
    print("Generated:", repr(output))
