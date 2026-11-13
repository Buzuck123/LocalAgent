"""DDGS search provider."""

import os

from ddgs import DDGS


ENGINES = (
    "auto",
    "all",
    "bing",
    "brave",
    "duckduckgo",
    "google",
    "grokipedia",
    "mojeek",
    "startpage",
    "yandex",
    "yahoo",
    "wikipedia",
)


def search(
    query: str,
    *,
    engine: str = "auto",
    limit: int = 5,
    region: str = "ru-ru",
    timeout: int = 15,
    safesearch: str = "moderate",
) -> list[dict]:
    """Run a text search and return normalized results."""

    if engine not in ENGINES:
        raise ValueError(
            f"Unsupported engine '{engine}'. "
            f"Available engines: {', '.join(ENGINES)}"
        )

    proxy = (
        os.environ.get("DDGS_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )

    client = DDGS(proxy=proxy, timeout=timeout)

    raw_results = client.text(
        query,
        backend=engine,
        region=region,
        safesearch=safesearch,
        max_results=limit,
    )

    results = []

    for item in raw_results or []:
        results.append({
            "title": str(item.get("title") or "").strip(),
            "url": str(item.get("href") or item.get("url") or "").strip(),
            "description": str(item.get("body") or item.get("description") or "").strip(),
        })

    return results
