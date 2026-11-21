
from src.data.wikipedia import fetch_article

article = fetch_article("Neural network")

print("Article:", article["title"])
print("License:", article["license"])
print("Characters:", len(article["text"]))

print("\nFirst 1000 characters:")
print(article["text"][:1000])
