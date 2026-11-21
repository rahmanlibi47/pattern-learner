
import hashlib
import json
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
    "geography",
    "biology",
    "physics",
    "computing",
    "astronomy",
]


def request_json(url: str, retries: int = 3) -> dict:
    """Make an API request with exponential backoff."""

    for attempt in range(retries):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=45) as response:
                return json.loads(
                    response.read().decode("utf-8")
                )

        except Exception:
            if attempt == retries - 1:
                raise

            delay = 2 ** attempt
            print(f"Request failed. Retrying in {delay}s...")
            time.sleep(delay)

    return {}


def search_articles(query: str, limit: int = 20) -> list[dict]:
    """Find Wikipedia articles matching a search query."""

    params = urlencode({
        "q": query,
        "limit": min(limit, 100),
    })

    url = f"{API_BASE}/search/page?{params}"

    try:
        data = request_json(url)
        return data.get("pages", [])

    except Exception as error:
        print(f"Search failed for '{query}': {error}")
        return []


def fetch_with_retries(title: str) -> dict | None:
    """Fetch an article, retrying transient failures."""

    for attempt in range(3):
        try:
            return fetch_article(title)

        except Exception as error:
            if attempt == 2:
                print(f"Skipping {title}: {error}")
                return None

            delay = 2 ** attempt
            print(
                f"Fetch failed for {title}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)

    return None


def is_relevant_title(title: str, topic: str) -> bool:
    """Avoid obvious search-result noise such as albums and journals."""
    normalized = title.casefold()
    excluded_terms = (
        "(disambiguation)", "(journal)", "journal", "album", "soundtrack",
        "film", "song", "single", "novel", "book i", "book ii",
    )
    if any(term in normalized for term in excluded_terms):
        return False
    topic_terms = set(topic.casefold().split())
    title_terms = set(normalized.replace("(", " ").replace(")", " ").split())
    # Accept exact topic matches, titles containing the topic, and specific
    # encyclopedia subtopics whose title includes the topic in parentheses.
    return bool(topic_terms & title_terms) or normalized == topic.casefold()


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
    max_article_chars: int = 5_000,
    max_per_topic: int = 2,
    on_article=None,
) -> list[dict]:
    """
    Discover and collect diverse Wikipedia articles in memory.
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

        if total_chars >= max_chars:
            break

        print(f"\nSearching: {topic}")

        results = search_articles(
            topic,
            limit=min(max_per_topic * 5, 100),
        )

        print(f"Found {len(results)} search results")

        topic_count = 0

        for result in results:
            if len(articles) >= max_articles:
                break

            if total_chars >= max_chars:
                break

            if topic_count >= max_per_topic:
                break

            title = result.get("title", "").strip()
            normalized_title = title.casefold()

            if not title or normalized_title in seen_titles:
                continue

            # Exclude Wikipedia disambiguation pages.
            if not is_relevant_title(title, topic):
                print(f"Skipping unrelated search result: {title}")
                continue

            seen_titles.add(normalized_title)

            # Be polite to Wikimedia's servers.
            time.sleep(1)

            article = fetch_with_retries(title)

            if article is None:
                continue

            text = article.get("text", "")
            license_info = article.get("license") or {}

            if not has_accepted_license(article):
                print(f"Skipping {title}: unsupported license")
                continue

            if len(text) < min_chars:
                print(f"Skipping {title}: too short")
                continue

            remaining = max_chars - total_chars
            article_limit = min(max_article_chars, remaining)

            text = text[:article_limit]

            # Avoid ending in the middle of a paragraph.
            if "\n" in text:
                text = text.rsplit("\n", 1)[0]

            if len(text) < min_chars:
                continue

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
                "topic": topic,
            }

            articles.append(collected)
            if on_article is not None:
                on_article(collected)

            total_chars += len(text)
            topic_count += 1

            print(
                f"Collected: {article['title']} "
                f"({len(text)} characters)"
            )

    print("\nCollection complete")
    print("Articles:", len(articles))
    print("Characters:", total_chars)

    return articles
