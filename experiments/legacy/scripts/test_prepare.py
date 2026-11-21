
from src.data.collector import collect_articles
from src.data.prepare import prepare_articles

articles = collect_articles(
    max_articles=15,
    max_chars=50_000,
    max_per_topic=2,
)

datasets = prepare_articles(articles)

print("\nDATASET SPLIT")
print("=" * 40)

for name, split in datasets.items():
    characters = sum(
        len(article["text"]) for article in split
    )

    print(f"{name}:")
    print("  Articles:", len(split))
    print("  Characters:", characters)

    for article in split:
        print("  -", article["title"])
