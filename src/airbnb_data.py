"""Small, testable utilities shared by the acquisition and audit scripts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import urlparse


SOURCE_PAGE_URL = "https://insideairbnb.com/get-the-data/"
DATA_DICTIONARY_URL = "https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?usp=sharing"
MEXICO_CURRENCY_SOURCE_URL = "https://www.ordenjuridico.gob.mx/Documentos/Federal/html/wo14412.html"
MEXICO_CITY_PATH = "/mexico/df/mexico-city/"
SNAPSHOT_DATE_RE = re.compile(r"(?:^|/|_)(20\d{2}-\d{2}-\d{2})(?:/|_|$)")
LISTINGS_GZ_RE = re.compile(r"/data/listings\.csv\.gz(?:\?.*)?$", re.IGNORECASE)


@dataclass(frozen=True)
class SnapshotLink:
    """A source link whose date was read from the official URL path."""

    snapshot_date: str
    url: str


def snapshot_date_from_url(url: str) -> str | None:
    """Return the ISO snapshot date embedded in a data URL, if present."""

    match = SNAPSHOT_DATE_RE.search(url)
    return match.group(1) if match else None


def discover_snapshot_links(html: str, city_path: str = MEXICO_CITY_PATH) -> list[SnapshotLink]:
    """Extract detailed Mexico City listings links from an official HTML page.

    The function intentionally only returns links present in the page source. It
    does not construct dates or URLs, which prevents silently using an unverified
    archive path.
    """

    pattern = re.compile(
        r"<a\b[^>]*?href\s*=\s*(['\"])(.*?)\1[^>]*>",
        re.IGNORECASE | re.DOTALL,
    )
    found: dict[str, SnapshotLink] = {}
    for match in pattern.finditer(html):
        raw_url = unescape(match.group(2)).strip()
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if "data.insideairbnb.com" not in parsed.netloc:
            continue
        if city_path not in parsed.path or not LISTINGS_GZ_RE.search(parsed.path):
            continue
        snapshot_date = snapshot_date_from_url(parsed.path)
        if snapshot_date:
            found[raw_url] = SnapshotLink(snapshot_date=snapshot_date, url=raw_url)
    return sorted(found.values(), key=lambda item: item.snapshot_date)


def _is_missing(value: Any) -> bool:
    if value is None or value.__class__.__name__ in {"NAType", "NaTType"}:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def parse_price(value: Any) -> float | None:
    """Parse common listing price strings without assigning a currency.

    This function parses the numeric amount only. A dollar sign is deliberately
    not interpreted as MXN because the source data must establish the currency.
    """

    if _is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else None

    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    numeric = re.sub(r"[^0-9.,+-]", "", text)
    if not re.search(r"\d", numeric):
        return None

    if "," in numeric and "." in numeric:
        if numeric.rfind(",") > numeric.rfind("."):
            numeric = numeric.replace(".", "").replace(",", ".")
        else:
            numeric = numeric.replace(",", "")
    elif "," in numeric:
        tail = numeric.rsplit(",", 1)[1]
        numeric = numeric.replace(",", "") if len(tail) == 3 else numeric.replace(",", ".")
    elif numeric.count(".") > 1:
        numeric = numeric.replace(".", "")

    try:
        amount = float(numeric)
    except ValueError:
        return None
    if negative:
        amount = -amount
    return amount if math.isfinite(amount) else None


def currency_marker(value: Any) -> str | None:
    """Extract an explicit symbol/code marker, without guessing its meaning."""

    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    for code in ("MXN", "USD", "CAD", "EUR", "GBP", "JPY", "CNY", "BRL"):
        if re.search(rf"\b{code}\b", upper):
            return code
    for symbol, label in (("MX$", "MX$"), ("US$", "US$"), ("€", "€"), ("£", "£"), ("¥", "¥"), ("₹", "₹")):
        if symbol in text:
            return label
    if "$" in text:
        return "symbol:$ (ambiguous)"
    return None
