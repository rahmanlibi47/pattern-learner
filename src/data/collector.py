
import hashlib
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.data.wikipedia import fetch_article, USER_AGENT


API_BASE = "https://en.wikipedia.org/w/rest.php/v1"

DEFAULT_TOPICS = [
    "science",
    "nature",
    "technology",
    "history",
    "mathematics",
]


def search_articles(query: str, limit: int = 20) -> list[dict]:
    """Find Wikipedia articles matching a search query."""

    params = urlencode({
        "q": query,
        "limit": min(limit, 100),
    })

    url = f"{API_BASE}/search/page?{params}"

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            import json
            data = json.loads(response.read().decode("utf-8"))

    except Exception as error:
        print(f"Search failed for '{query}': {error}")
        return []

    return data.get("pages", [])


def has_accepted_license(article: dict) -> bool:
    """Accept recognized Creative Commons BY-SA article licenses."""

    license_info = article.get("license") or {}
    license_url = license_info.get("url", "").lower()

    return (
        "creativecommons.org/licenses/by-sa/" in license_url
        and (
            "/3.0" in license_url
            or "/4.0" in license_url
        )
    )


def collect_articles(
    topics: list[str] | None = None,
    max_articles: int = 20,
    max_chars: int = 100_000,
    min_chars: int = 500,
) -> list[dict]:
    """
    Search Wikipedia and collect article text in memory.

    Returns article dictionaries containing text and attribution.
    """

    if topics is None:
        topics = DEFAULT_TOPICS

    articles = []
    seen_titles = set()
    seen_hashes = set()
    total_chars = 0

    for topic in topics:
        if len(articles) >= max_articles:
            break

        print(f"\nSearching: {topic}")

        results = search_articles(
            topic,
            limit=min(max_articles * 2, 100),
        )

        print(f"Found {len(results)} search results")

        for result in results:
            if len(articles) >= max_articles:
                break

            title = result.get("title", "").strip()
            normalized_title = title.casefold()

            if not title or normalized_title in seen_titles:
                continue

            seen_titles.add(normalized_title)

            # Be polite to Wikimedia's servers.
            time.sleep(0.5)

            try:
                article = fetch_article(title)
            except Exception as error:
                print(f"Skipping {title}: {error}")
                continue

            text = article["text"]
            license_info = article.get("license") or {}

            if not has_accepted_license(article):
                print(f"Skipping {title}: unsupported license")
                continue

            if len(text) < min_chars:
                print(f"Skipping {title}: too short")
                continue

            remaining = max_chars - total_chars

            if remaining <= 0:
                break

            # Keep the corpus within the configured character budget.
            if len(text) > remaining:
                text = text[:remaining]

            content_hash = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

            if content_hash in seen_hashes:
                continue

            seen_hashes.add(content_hash)

            collected = {
                "title": article["title"],
                "text": text,
                "url": article.get("url"),
                "license": license_info.get("title"),
                "license_url": license_info.get("url"),
            }

            articles.append(collected)
            total_chars += len(text)

            print(
                f"Collected: {article['title']} "
                f"({len(text)} characters)"
            )

            if total_chars >= max_chars:
                break

    print("\nCollection complete")
    print("Articles:", len(articles))
    print("Characters:", total_chars)

    return articles
