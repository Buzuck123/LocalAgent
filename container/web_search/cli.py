"""Command-line interface for web search."""

import argparse
import json
import sys

from web_search.providers.ddgs import ENGINES, search
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="web-search",
        description="Search the web from the LocalAgent container.",
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default="auto",
        help="Search backend (default: auto)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)",
    )
    parser.add_argument(
        "--region",
        default="ru-ru",
        help="Search region and language, e.g. ru-ru or us-en",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Request timeout in seconds (default: 15)",
    )
    parser.add_argument(
        "--safe-search",
        choices=("on", "moderate", "off"),
        default="moderate",
        help="Safe search level (default: moderate)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.query.strip():
        parser.error("query must not be empty")

    if not 1 <= args.limit <= 50:
        parser.error("--limit must be between 1 and 50")

    if not 1 <= args.timeout <= 120:
        parser.error("--timeout must be between 1 and 120 seconds")

    try:
        results = search(
            args.query.strip(),
            engine=args.engine,
            limit=args.limit,
            region=args.region,
            timeout=args.timeout,
            safesearch=args.safe_search,
        )
    except Exception as exc:
        if isinstance(exc, RatelimitException):
            message = f"Search rate limit reached: {exc}"
        elif isinstance(exc, TimeoutException):
            message = f"Search timed out: {exc}"
        elif isinstance(exc, DDGSException):
            message = f"Search failed (DDGSException): {exc}"
        else:
            message = f"Search failed ({type(exc).__name__}): {exc}"

        if args.json:
            print(json.dumps(
                {"query": args.query, "error": message, "results": []},
                ensure_ascii=False,
            ))
        else:
            print(message, file=sys.stderr)

        return 2

    payload = {
        "query": args.query,
        "engine": args.engine,
        "count": len(results),
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not results:
        print("No results found.")
        return 0

    for index, item in enumerate(results, start=1):
        print(f"{index}. {item['title'] or '(untitled)'}")
        print(f"   {item['url']}")
        if item["description"]:
            print(f"   {item['description']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
