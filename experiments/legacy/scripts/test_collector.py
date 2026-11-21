
from src.data.collector import collect_articles

articles = collect_articles(
    max_articles=15,
    max_chars=50_000,
    max_article_chars=5_000,
    max_per_topic=2,
)

print("\nARTICLE SUMMARY")
print("=" * 50)

for article in articles:
    print(f"\nTitle: {article['title']}")
    print(f"Topic: {article['topic']}")
    print(f"Characters: {len(article['text'])}")
    print(f"License: {article['license']}")
    print(f"URL: {article['url']}")

print("\nTOTAL CHARACTERS:", sum(
    len(article["text"]) for article in articles
))
