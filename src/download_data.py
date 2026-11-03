"""Discover and download official Inside Airbnb detailed listings snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.airbnb_data import (
    SOURCE_PAGE_URL,
    SnapshotLink,
    discover_snapshot_links,
    snapshot_date_from_url,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "data_manifest.json"


def fetch_bytes(url: str, timeout: int = 60) -> tuple[bytes, str | None]:
    """Fetch bytes and return them with the server-reported content type."""

    request = Request(url, headers={"User-Agent": "data-science-airbnb-pricing/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers.get_content_type()


def discover_official_links(source_url: str = SOURCE_PAGE_URL) -> list[SnapshotLink]:
    html_bytes, _ = fetch_bytes(source_url)
    html = html_bytes.decode("utf-8", errors="replace")
    return discover_snapshot_links(html)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"El manifiesto debe ser una lista JSON: {path}")
    return payload


def write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_snapshot(link: SnapshotLink, raw_dir: Path, timeout: int = 120) -> dict[str, object]:
    """Download bytes unchanged and return auditable provenance metadata."""

    content, content_type = fetch_bytes(link.url, timeout=timeout)
    output_path = raw_dir / f"mexico_city_{link.snapshot_date}_listings.csv.gz"
    raw_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    temp_path.write_bytes(content)
    temp_path.replace(output_path)
    return {
        "source": "Inside Airbnb",
        "source_page": SOURCE_PAGE_URL,
        "snapshot_date": link.snapshot_date,
        "url": link.url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(output_path.relative_to(REPO_ROOT)),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_bytes(content),
        "content_type": content_type,
        "file_role": "detailed listings.csv.gz",
        "bytes_modified": False,
    }


def select_links(links: list[SnapshotLink], limit: int = 2) -> list[SnapshotLink]:
    """Select the most recent chronological links discovered on the source page."""

    unique = {link.url: link for link in links}
    return sorted(unique.values(), key=lambda item: item.snapshot_date)[-limit:]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=SOURCE_PAGE_URL)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--url",
        action="append",
        help="URL exacta verificada; se puede repetir. No se construyen URLs automáticamente.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
        manifest_path = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
        manifest = load_manifest(manifest_path)
        if args.url:
            links = []
            for url in args.url:
                snapshot_date = snapshot_date_from_url(url)
                if snapshot_date is None:
                    raise ValueError(f"No se encontró fecha ISO en la URL: {url}")
                links.append(SnapshotLink(snapshot_date=snapshot_date, url=url))
        else:
            discovered = discover_official_links(args.source_url)
            recorded = []
            for item in manifest:
                url = item.get("url")
                snapshot_date = item.get("snapshot_date") or snapshot_date_from_url(str(url))
                if isinstance(url, str) and isinstance(snapshot_date, str):
                    recorded.append(SnapshotLink(snapshot_date=snapshot_date, url=url))
            links = select_links(discovered + recorded, limit=2)

        if len(links) < 2:
            print(
                "Se requieren dos snapshots descargables y la página oficial solo expuso "
                f"{len(links)} enlace(s) detallado(s) en esta ejecución. "
                "No se inventó una URL de archivo; revise conectividad o use --url con enlaces verificados.",
                file=sys.stderr,
            )
            return 2

        for link in links:
            print(f"Descargando {link.snapshot_date}: {link.url}")
            entry = download_snapshot(link, raw_dir, timeout=args.timeout)
            manifest = [item for item in manifest if item.get("url") != link.url]
            manifest.append(entry)
            print(f"  {entry['path']} | {entry['size_bytes']} bytes | sha256={entry['sha256']}")
        write_manifest(manifest_path, sorted(manifest, key=lambda item: str(item.get("snapshot_date", ""))))
        print(f"Manifiesto actualizado: {manifest_path}")
        return 0
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"No se completó la descarga: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
