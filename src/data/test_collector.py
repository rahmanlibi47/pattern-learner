
from src.data.collector import collect_articles

articles = collect_articles(
    topics=["science", "nature", "technology"],
    max_articles=5,
    max_chars=15_000,
)

print("\nARTICLE SUMMARY")
print("=" * 40)

for article in articles:
    print(f"\nTitle: {article['title']}")
    print(f"Characters: {len(article['text'])}")
    print(f"License: {article['license']}")
    print(f"URL: {article['url']}")

if articles:
    print("\nSAMPLE TEXT")
    print("=" * 40)
    print(articles[0]["text"][:500])
