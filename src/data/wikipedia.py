
import json
import os
import re
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


API_BASE = "https://en.wikipedia.org/w/rest.php/v1"

# Replace this with your actual GitHub repository URL.
USER_AGENT = os.getenv(
    "WIKIPEDIA_USER_AGENT",
    "PatternLearner/0.2 (self-hosted educational research project)",
)


class ArticleParser(HTMLParser):
    """Extract readable text from article HTML."""

    BLOCK_TAGS = {
        "p", "h1", "h2", "h3", "h4",
        "li", "div", "section", "br"
    }

    SKIP_TAGS = {
        "script", "style", "nav", "footer",
        "header", "sup"
    }

    VOID_TAGS = {
        "area", "base", "br", "col", "embed",
        "hr", "img", "input", "link", "meta",
        "param", "source", "track", "wbr"
    }

    SKIP_CLASSES = {
    "mw-editsection",
    "navbox",
    "vertical-navbox",
    "sidebar",
    "metadata",
    "ambox",
    "reflist",
    "hatnote",
    "shortdescription",
    "catlinks",
    "toc",
    "sistersitebox",
}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        classes = set(attrs.get("class", "").split())

        parent_skipped = (
            self.stack[-1][1] if self.stack else False
        )

        skipped = (
            parent_skipped
            or tag in self.SKIP_TAGS
            or bool(classes & self.SKIP_CLASSES)
        )

        if not skipped and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

        if tag not in self.VOID_TAGS:
            self.stack.append((tag, skipped))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                self.stack = self.stack[:index]
                break

        currently_skipped = (
            self.stack[-1][1] if self.stack else False
        )

        if not currently_skipped and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.stack or not self.stack[-1][1]:
            self.parts.append(data)

    def get_text(self):
        text = "".join(self.parts)

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)

        return text.strip()


def fetch_article(title: str) -> dict:
    """Fetch one Wikipedia article directly from the API."""

    encoded_title = quote(title.replace(" ", "_"), safe="")

    url = f"{API_BASE}/page/{encoded_title}/with_html"

    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))

    except HTTPError as error:
        raise RuntimeError(
            f"Wikipedia returned HTTP {error.code}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Could not connect to Wikipedia: {error.reason}"
        ) from error

    parser = ArticleParser()
    parser.feed(data["html"])

    return {
        "title": data["title"],
        "text": parser.get_text(),
        "license": data.get("license"),
        "url": data.get("html_url") or (
    f"https://en.wikipedia.org/wiki/{encoded_title}"
),
    }
