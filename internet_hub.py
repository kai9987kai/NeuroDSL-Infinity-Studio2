"""Internet utilities for fetching world events and extracting phrase signals."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


DEFAULT_HEADERS = {
    "User-Agent": "NeuroDSL-Infinity-Studio/1.0 (+https://localhost)",
    "Accept": "application/json, text/xml, application/xml, text/plain;q=0.9,*/*;q=0.8",
}

DEFAULT_WORLD_FEEDS = [
    {"source": "Reuters World", "url": "https://feeds.reuters.com/Reuters/worldNews", "region": "global"},
    {"source": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "region": "global"},
    {"source": "UN News", "url": "https://news.un.org/feed/subscribe/en/news/all/rss.xml", "region": "global"},
    {"source": "AP Top News", "url": "https://apnews.com/hub/ap-top-news?output=1", "region": "global"},
]

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "over",
    "after",
    "before",
    "under",
    "about",
    "update",
    "world",
    "global",
    "news",
    "said",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_text(url: str, timeout: float = 8.0) -> str:
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def fetch_json(url: str, timeout: float = 8.0) -> dict:
    text = fetch_text(url, timeout=timeout)
    return json.loads(text)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _parse_rss_items(xml_text: str, source: str, region: str) -> list[dict]:
    events: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return events

    items = root.findall(".//item")
    if items:
        for item in items:
            events.append(
                {
                    "title": (item.findtext("title") or "").strip(),
                    "summary": _strip_html(item.findtext("description") or ""),
                    "url": (item.findtext("link") or "").strip(),
                    "published_at": (
                        item.findtext("pubDate")
                        or item.findtext("updated")
                        or item.findtext("published")
                        or ""
                    ).strip(),
                    "source": source,
                    "region": region,
                }
            )
        return events

    # Atom feed fallback.
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall(".//atom:entry", ns)
    for entry in entries:
        link_el = entry.find("atom:link", ns)
        events.append(
            {
                "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
                "summary": _strip_html(entry.findtext("atom:summary", default="", namespaces=ns) or ""),
                "url": (link_el.get("href") if link_el is not None else "") or "",
                "published_at": (
                    entry.findtext("atom:updated", default="", namespaces=ns)
                    or entry.findtext("atom:published", default="", namespaces=ns)
                    or ""
                ).strip(),
                "source": source,
                "region": region,
            }
        )
    return events


def get_builtin_world_events() -> list[dict]:
    ts = _now_iso()
    return [
        {
            "title": "Global AI infrastructure investment activity remains elevated",
            "summary": "Multiple regions continue to announce compute and data-center expansion plans.",
            "url": "",
            "published_at": ts,
            "source": "builtin",
            "region": "global",
        },
        {
            "title": "Energy transition and climate adaptation initiatives expand across regions",
            "summary": "Public and private sectors report additional adaptation and resilience projects.",
            "url": "",
            "published_at": ts,
            "source": "builtin",
            "region": "global",
        },
        {
            "title": "Cross-border semiconductor and advanced manufacturing partnerships continue",
            "summary": "New agreements emphasize supply-chain reliability and multi-country collaboration.",
            "url": "",
            "published_at": ts,
            "source": "builtin",
            "region": "global",
        },
    ]


def fetch_world_events(
    max_items: int = 50,
    include_network: bool = True,
    feeds: list[dict] | None = None,
    timeout: float = 8.0,
) -> list[dict]:
    events: list[dict] = []
    events.extend(get_builtin_world_events())
    if include_network:
        for feed in feeds or DEFAULT_WORLD_FEEDS:
            source = str(feed.get("source", "unknown"))
            url = str(feed.get("url", "")).strip()
            region = str(feed.get("region", "global"))
            if not url:
                continue
            try:
                xml_text = fetch_text(url, timeout=timeout)
                events.extend(_parse_rss_items(xml_text, source=source, region=region))
            except (urllib.error.URLError, TimeoutError, ValueError):
                continue
            except Exception:
                continue

    dedup: list[dict] = []
    seen: set[str] = set()
    for event in events:
        key = "|".join(
            [
                str(event.get("source", "")),
                str(event.get("title", "")),
                str(event.get("published_at", "")),
            ]
        ).lower()
        if not event.get("title") or key in seen:
            continue
        seen.add(key)
        dedup.append(event)
        if len(dedup) >= max(1, int(max_items)):
            break
    return dedup


def extract_keyphrases(texts: list[str], top_k: int = 50) -> list[dict]:
    freq: dict[str, int] = {}
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-']{2,}", (text or "").lower()):
            if token in STOPWORDS:
                continue
            freq[token] = freq.get(token, 0) + 1
    sorted_terms = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {"language": "en", "phrase": phrase, "frequency": count, "source": "events"}
        for phrase, count in sorted_terms[: max(1, int(top_k))]
    ]
