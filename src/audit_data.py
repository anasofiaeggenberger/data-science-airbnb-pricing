"""Audit downloaded Inside Airbnb detailed listings snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.airbnb_data import (
    DATA_DICTIONARY_URL,
    MEXICO_CURRENCY_SOURCE_URL,
    currency_marker,
    parse_price,
    snapshot_date_from_url,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_REPORT = REPO_ROOT / "docs" / "data_audit.md"
DEFAULT_RESULTS = REPO_ROOT / "reports" / "audit_results.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blank_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    return text.isna() | text.str.strip().eq("")


def _read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, compression="infer", nrows=0).columns)


def audit_file(path: Path, chunksize: int = 50_000) -> dict[str, object]:
    """Audit a detailed listings file in chunks, without modifying it."""

    columns = _read_header(path)
    try:
        relative_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        relative_path = str(path)
    result: dict[str, object] = {
        "path": relative_path,
        "snapshot_date": snapshot_date_from_url(f"/{path.name}/"),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "columns": columns,
        "row_count": 0,
        "id_column_present": "id" in columns,
        "id_null_count": None,
        "id_unique_count": None,
        "id_duplicate_count": None,
        "price_column_present": "price" in columns,
        "price_null_count": None,
        "price_parseable_count": None,
        "price_invalid_count": None,
        "price_format_counts": {},
        "currency_column_present": "currency" in columns,
        "currency_values": {},
        "price_currency_markers": {},
        "price_examples": [],
        "issues": [],
    }
    issues: list[str] = []
    if "id" not in columns:
        issues.append("Falta la columna obligatoria id.")
    if "price" not in columns:
        issues.append("Falta la columna obligatoria price.")

    id_seen: set[str] = set()
    id_null_count = 0
    price_null_count = 0
    price_parseable_count = 0
    price_invalid_count = 0
    price_format_counts: Counter[str] = Counter()
    currency_values: Counter[str] = Counter()
    price_currency_markers: Counter[str] = Counter()
    price_examples: list[str] = []

    reader = pd.read_csv(path, compression="infer", dtype="string", chunksize=chunksize)
    for chunk in reader:
        result["row_count"] = int(result["row_count"]) + len(chunk)
        if "id" in chunk:
            id_blank = _blank_mask(chunk["id"])
            id_null_count += int(id_blank.sum())
            for value in chunk.loc[~id_blank, "id"].astype(str):
                id_seen.add(value)
        if "price" in chunk:
            price_blank = _blank_mask(chunk["price"])
            price_null_count += int(price_blank.sum())
            for is_blank, value in zip(price_blank.tolist(), chunk["price"].tolist()):
                if len(price_examples) < 10 and not is_blank:
                    price_examples.append(str(value))
                parsed = parse_price(value)
                if is_blank:
                    price_format_counts["missing"] += 1
                elif parsed is None:
                    price_format_counts["invalid"] += 1
                    price_invalid_count += 1
                else:
                    price_format_counts["parseable"] += 1
                    price_parseable_count += 1
                marker = currency_marker(value)
                if marker:
                    price_currency_markers[marker] += 1
        if "currency" in chunk:
            for value in chunk["currency"].tolist():
                if value is not None and str(value).strip():
                    currency_values[str(value).strip()] += 1

    id_non_null_count = int(result["row_count"]) - id_null_count if "id" in columns else 0
    id_duplicate_count = id_non_null_count - len(id_seen) if "id" in columns else None
    result.update(
        {
            "id_null_count": id_null_count if "id" in columns else None,
            "id_unique_count": len(id_seen) if "id" in columns else None,
            "id_duplicate_count": id_duplicate_count,
            "price_null_count": price_null_count if "price" in columns else None,
            "price_parseable_count": price_parseable_count if "price" in columns else None,
            "price_invalid_count": price_invalid_count if "price" in columns else None,
            "price_format_counts": dict(price_format_counts),
            "currency_values": dict(currency_values),
            "price_currency_markers": dict(price_currency_markers),
            "price_examples": price_examples,
        }
    )
    if "id" in columns and (id_null_count or id_duplicate_count):
        issues.append("El identificador id tiene nulos o duplicados.")
    if "price" in columns and price_invalid_count:
        issues.append("Hay valores de price no parseables.")
    if "price" in columns and "currency" not in columns:
        issues.append("No existe columna currency; la moneda se documenta con la definición oficial de price.")
    if "price" in columns and price_currency_markers.get("symbol:$ (ambiguous)"):
        issues.append("El símbolo $ es un artefacto técnico del export según el diccionario oficial; no se usó como evidencia independiente.")
    result["issues"] = issues
    return result


def ids_from_file(path: Path, chunksize: int = 50_000) -> set[str]:
    """Read only the id column for an exact intersection comparison."""

    columns = _read_header(path)
    if "id" not in columns:
        return set()
    ids: set[str] = set()
    for chunk in pd.read_csv(path, compression="infer", usecols=["id"], dtype="string", chunksize=chunksize):
        valid = chunk["id"][~_blank_mask(chunk["id"])]
        ids.update(valid.astype(str).tolist())
    return ids


def compare_snapshots(paths: Iterable[Path]) -> dict[str, object] | None:
    paths = list(paths)
    if len(paths) != 2:
        return None
    first_ids, second_ids = (ids_from_file(path) for path in paths)
    try:
        first_name = str(paths[0].relative_to(REPO_ROOT))
        second_name = str(paths[1].relative_to(REPO_ROOT))
    except ValueError:
        first_name, second_name = (str(path) for path in paths)
    intersection = first_ids & second_ids
    return {
        "snapshot_a": first_name,
        "snapshot_b": second_name,
        "ids_a": len(first_ids),
        "ids_b": len(second_ids),
        "intersection_count": len(intersection),
        "only_a_count": len(first_ids - second_ids),
        "only_b_count": len(second_ids - first_ids),
        "intersection_rate_of_a": len(intersection) / len(first_ids) if first_ids else None,
        "intersection_rate_of_b": len(intersection) / len(second_ids) if second_ids else None,
    }


def render_report(results: dict[str, object]) -> str:
    files = results.get("files", [])
    lines = [
        "# Auditoría inicial de datos",
        "",
        "Proyecto: `data-science-airbnb-pricing` — Ciudad de México.",
        "",
        f"Ejecución UTC: `{results['generated_at_utc']}`.",
        "",
        "## Alcance",
        "",
        "Esta auditoría valida infraestructura y calidad básica del archivo detallado "
        "`listings.csv.gz`. No contiene EDA profundo, entrenamiento ni resultados de modelos.",
        "",
        "## Procedencia",
        "",
        "Fuente: [Inside Airbnb — Get the Data](https://insideairbnb.com/get-the-data/). "
        "Las URLs exactas, fechas de descarga, tamaños y SHA-256 están en `docs/data_manifest.json`.",
        f"El [diccionario oficial de datos]({DATA_DICTIONARY_URL}) define `price` como precio diario en moneda local y documenta el símbolo `$` como artefacto técnico del export.",
        f"La [Ley Monetaria de México]({MEXICO_CURRENCY_SOURCE_URL}) establece el peso como unidad del sistema monetario; por eso el objetivo monetario del proyecto se documenta como MXN.",
        "",
        "## Estado de los archivos",
        "",
    ]
    if not files:
        lines += [
            "No hay snapshots descargados en `data/raw` en esta ejecución.",
            "",
            "Quedan pendientes: descarga de dos snapshots oficiales, tamaños, SHA-256, columnas, filas, "
            "unicidad/nulos de `id`, formato y moneda de `price`, y coincidencias de IDs.",
            "",
        ]
    else:
        lines += [
            "| Archivo | Filas | Columnas | id nulos | id únicos | id duplicados | price nulos | price parseable | price inválidos | SHA-256 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for item in files:
            lines.append(
                f"| `{item['path']}` | {item['row_count']} | {len(item['columns'])} | "
                f"{item['id_null_count']} | {item['id_unique_count']} | {item['id_duplicate_count']} | "
                f"{item['price_null_count']} | {item['price_parseable_count']} | {item['price_invalid_count']} | "
                f"`{item['sha256']}` |"
            )
        lines.append("")
        for item in files:
            lines += [
                f"### `{item['path']}`",
                "",
                f"- Snapshot: `{item.get('snapshot_date') or 'no identificado en el nombre'}`",
                f"- Tamaño: `{item['size_bytes']}` bytes",
                f"- Columnas: `{', '.join(item['columns'])}`",
                f"- Columna `currency` presente: `{item['currency_column_present']}`",
                f"- Formato de `price`: `{item['price_format_counts']}`",
                f"- Marcadores de moneda en `price`: `{item['price_currency_markers']}`",
                f"- Valores de columna `currency`: `{item['currency_values']}`",
                "- Moneda interpretada: `MXN`, por la definición oficial de moneda local y el alcance geográfico Mexico City; el código MXN no aparece explícito en el CSV.",
                f"- Ejemplos no interpretados como evidencia de moneda: `{item['price_examples'][:5]}`",
                f"- Hallazgos: `{item['issues'] or 'ninguno en esta auditoría básica'}`",
                "",
            ]

    comparison = results.get("comparison")
    lines += ["## Comparación entre snapshots", ""]
    if comparison is None:
        lines += [
            "Pendiente: se necesitan exactamente dos archivos detallados auditados para calcular la intersección de IDs.",
            "",
        ]
    else:
        lines += [
            f"- IDs en A: `{comparison['ids_a']}`; IDs en B: `{comparison['ids_b']}`.",
            f"- Intersección: `{comparison['intersection_count']}`.",
            f"- Solo A: `{comparison['only_a_count']}`; solo B: `{comparison['only_b_count']}`.",
            f"- Cobertura de A en B: `{comparison['intersection_rate_of_a']}`; de B en A: `{comparison['intersection_rate_of_b']}`.",
            "",
        ]

    lines += [
        "## Decisiones de framing y fuga de información",
        "",
        "El precio se recomendará para un anfitrión nuevo: no se deben usar reseñas, calificación, "
        "número de reseñas, superhost, tasa de respuesta, aceptación o cancelación si no están "
        "disponibles al publicar el anuncio. También se revisarán como potencial fuga variables "
        "calculadas después de publicar, como disponibilidad futura observada, precio histórico, "
        "métricas de ocupación, ingresos, reservas y cualquier señal derivada de reviews.",
        "",
        "En estas columnas concretas habrá que excluir o justificar `availability_30/60/90/365`, "
        "`availability_eoy`, `calendar_updated`, `calendar_last_scraped`, `estimated_occupancy_l365d`, "
        "`estimated_revenue_l365d`, `price_quote_*`, `number_of_reviews*`, `review_scores_*`, "
        "`first_review`, `last_review`, `reviews_per_month`, `host_is_superhost`, métricas de respuesta/aceptación, "
        "`calculated_host_listings_count*` y metadatos de scrape como `scrape_id`, `last_scraped` y `source`.",
        "",
        "El diccionario oficial define `price` como moneda local y el símbolo `$` como artefacto técnico. "
        "Para estos snapshots de Mexico City, esa definición sustenta reportar MXN; el CSV no contiene una columna de código de moneda.",
        "",
        "## Limitaciones de esta ejecución",
        "",
        "La descarga y esta auditoría deben repetirse si se reemplazan los snapshots. "
        "Los bytes crudos no se transforman; cualquier derivado futuro debe quedar en `data/processed`.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="Dos listings.csv.gz; por defecto, busca en data/raw")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = args.files or sorted(args.raw_dir.glob("*listings.csv.gz"))
    paths = [path if path.is_absolute() else REPO_ROOT / path for path in paths]
    results: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "comparison": None,
    }
    for path in paths:
        if not path.exists():
            raise SystemExit(f"No existe el archivo: {path}")
        results["files"].append(audit_file(path))
    results["comparison"] = compare_snapshots(paths)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(results), encoding="utf-8")
    print(render_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
