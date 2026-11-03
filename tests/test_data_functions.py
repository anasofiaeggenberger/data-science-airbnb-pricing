from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from src.airbnb_data import SnapshotLink, currency_marker, discover_snapshot_links, parse_price
from src.audit_data import audit_file, compare_snapshots
from src.download_data import select_links
from src.preparation import (
    build_temporal_split_manifest,
    feature_policy,
    parse_price_mxn,
    prepare_development_dataset,
    price_status,
)


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


def test_price_status_and_parse_price_mxn_reject_invalid_targets() -> None:
    assert price_status("$1,250") == "valid"
    assert parse_price_mxn("$1,250") == 1250.0
    assert price_status("") == "missing"
    assert price_status("not available") == "invalid"
    assert price_status("$0") == "zero_or_negative"
    assert pd.isna(parse_price_mxn("$0"))


def test_feature_policy_marks_temporal_candidates_optional_and_id_prohibited() -> None:
    decisions = {
        entry["column"]: entry["decision"]
        for entry in feature_policy(
            ["host_identity_verified", "host_has_profile_pic", "minimum_nights", "maximum_nights", "id"]
        )
    }
    assert decisions["host_identity_verified"] == "Permitida, opcional"
    assert decisions["host_has_profile_pic"] == "Permitida, opcional"
    assert decisions["minimum_nights"] == "Permitida, opcional"
    assert decisions["maximum_nights"] == "Permitida, opcional"
    assert decisions["id"] == "Prohibida"


def test_temporal_split_counts_new_ids_and_excludes_invalid_prices(tmp_path: Path) -> None:
    first = tmp_path / "mexico_city_2026-06-15_listings.csv.gz"
    second = tmp_path / "mexico_city_2026-09-23_listings.csv.gz"
    _write_gzip_csv(first, [("1", "$1,000", "Entire home/apt"), ("2", "$2,500", "Private room")])
    _write_gzip_csv(
        second,
        [
            ("2", "$2,500", "Private room"),
            ("3", "$3,000", "Entire home/apt"),
            ("4", "", "Entire home/apt"),
            ("5", "$0", "Private room"),
        ],
    )

    manifest = build_temporal_split_manifest(first, second, output_path=tmp_path / "split.json")

    assert manifest["test"]["new_id_count"] == 3
    assert manifest["test"]["eligible_new_id_count"] == 1
    assert manifest["test"]["excluded_new_id_count"] == 2
    assert manifest["test"]["excluded_new_by_price_status"] == {
        "missing": 1,
        "invalid": 0,
        "zero_or_negative": 1,
    }
    assert "price_status_counts" not in manifest["test"]
    assert manifest["test"]["records_saved"] is False
    assert manifest["test"]["ids_saved"] is False


def test_cleaning_uses_june_rules_and_keeps_traceability(tmp_path: Path) -> None:
    input_path = tmp_path / "mexico_city_2026-06-15_listings.csv.gz"
    raw = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5"],
            "price": ["$100", "$200", "", "$0", "$300"],
            "accommodates": [2, None, 10, 4, 6],
            "room_type": ["Private room", "Private room", "Shared room", "Shared room", "Hotel room"],
            "amenities": ["['Wifi', 'Kitchen']", "[]", "[]", "[]", "['Wifi']"],
        }
    )
    raw.to_csv(input_path, index=False, compression="gzip")

    prepared = prepare_development_dataset(
        input_path=input_path,
        output_path=tmp_path / "development_clean.csv",
        cleaning_log_path=tmp_path / "cleaning_log.json",
        feature_policy_path=tmp_path / "feature_policy.md",
        min_category_count=2,
    )

    assert len(prepared.data) == 3
    assert "id" not in prepared.data.columns
    assert prepared.data["price_mxn"].tolist() == [100.0, 200.0, 300.0]
    assert prepared.data["accommodates"].tolist() == [2.0, 4.0, 6.0]
    assert set(prepared.data["room_type"]) == {"Private room", "Other"}
    assert prepared.cleaning_log["rows"] == {
        "input": 5,
        "after_valid_id_and_price": 3,
        "output": 3,
        "rows_removed_total": 2,
    }
    assert prepared.cleaning_log["price_diagnostics_development_only"]["extremes_removed"] == 0
