
import random
import re


def normalize_text(text: str) -> str:
    """Normalize English text for character-level learning."""

    text = text.lower()

    # Keep English letters, spaces and basic punctuation.
    text = re.sub(r"[^a-z .,!?;:'-]", " ", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_articles(
    articles: list[dict],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split complete articles, not individual characters."""

    articles = list(articles)

    random.Random(seed).shuffle(articles)

    total = len(articles)

    if total < 3:
        raise ValueError(
            "At least 3 articles are required for train/validation/test."
        )

    train_count = max(1, int(total * train_ratio))
    validation_count = max(
    1,
    int(total * validation_ratio + 0.5)
)

    # Ensure every split gets at least one article.
    if train_count + validation_count >= total:
        train_count = total - 2
        validation_count = 1

    train = articles[:train_count]
    validation = articles[
        train_count:train_count + validation_count
    ]
    test = articles[train_count + validation_count:]

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def prepare_articles(articles: list[dict]) -> dict:
    """Normalize and split collected articles."""

    normalized = []

    for article in articles:
        text = normalize_text(article["text"])

        if len(text) < 500:
            continue

        normalized.append({
            **article,
            "text": text,
        })

    return split_articles(normalized)
