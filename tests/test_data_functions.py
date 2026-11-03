from __future__ import annotations

import gzip
from pathlib import Path

from src.airbnb_data import SnapshotLink, currency_marker, discover_snapshot_links, parse_price
from src.audit_data import audit_file, compare_snapshots
from src.download_data import select_links


def test_parse_price_keeps_currency_out_of_numeric_parsing() -> None:
    assert parse_price("$1,234.50") == 1234.5
    assert parse_price("MX$ 2,500") == 2500.0
    assert parse_price("1.234,56") == 1234.56
    assert parse_price("not available") is None
    assert currency_marker("$1,234.50") == "symbol:$ (ambiguous)"


def test_discover_snapshot_links_only_returns_detailed_city_links() -> None:
    html = """
    <a href="https://data.insideairbnb.com/mexico/df/mexico-city/2026-06-15/data/listings.csv.gz">Detailed</a>
    <a href="https://data.insideairbnb.com/mexico/df/mexico-city/2026-09-23/data/listings.csv.gz">Detailed</a>
    <a href="https://data.insideairbnb.com/mexico/df/mexico-city/2026-09-23/data/listings.csv">Summary</a>
    <a href="https://data.insideairbnb.com/mexico/df/mexico-city/2026-09-23/data/calendar.csv.gz">Calendar</a>
    """
    links = discover_snapshot_links(html)
    assert [link.snapshot_date for link in links] == ["2026-06-15", "2026-09-23"]
    assert all(link.url.endswith("listings.csv.gz") for link in links)


def test_select_links_deduplicates_registered_urls() -> None:
    links = [
        SnapshotLink("2026-06-15", "https://example.invalid/old"),
        SnapshotLink("2026-09-23", "https://example.invalid/new"),
        SnapshotLink("2026-09-23", "https://example.invalid/new"),
    ]
    selected = select_links(links)
    assert [link.snapshot_date for link in selected] == ["2026-06-15", "2026-09-23"]


def _write_gzip_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,price,room_type\n")
        for listing_id, price, room_type in rows:
            handle.write(f'{listing_id},"{price}","{room_type}"\n')


def test_audit_and_snapshot_intersection(tmp_path: Path) -> None:
    first = tmp_path / "mexico_city_2026-06-15_listings.csv.gz"
    second = tmp_path / "mexico_city_2026-09-23_listings.csv.gz"
    _write_gzip_csv(first, [("1", "$1,000", "Entire home/apt"), ("2", "$2,500", "Private room")])
    _write_gzip_csv(second, [("2", "$2,500", "Private room"), ("3", "$3,000", "Entire home/apt")])

    first_audit = audit_file(first)
    assert first_audit["row_count"] == 2
    assert first_audit["id_unique_count"] == 2
    assert first_audit["price_parseable_count"] == 2
    assert first_audit["price_currency_markers"]["symbol:$ (ambiguous)"] == 2

    comparison = compare_snapshots([first, second])
    assert comparison is not None
    assert comparison["intersection_count"] == 1
