"""Tablas por segmento con tamaños de muestra, para la interpretación de errores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import QUANTILE_LEVELS, SELECTION_METRIC, evaluate_quantile_predictions
from src.modeling_data import load_development
from src.preparation import REPO_ROOT


QUANTILE_COLUMNS = sorted(QUANTILE_LEVELS, key=lambda column: QUANTILE_LEVELS[column])
SEGMENT_COLUMNS = ("neighbourhood_cleansed", "room_type", "price_band")

# Debajo de este tamaño una conclusión por segmento no tiene evidencia suficiente.
MIN_SEGMENT_SIZE = 30

REPORTED_METRICS = [
    SELECTION_METRIC,
    "mae_recommended",
    "mae_median",
    "coverage",
    "mean_interval_width",
]

DEFAULT_PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "validation_predictions.csv"
DEFAULT_TABLES_PATH = REPO_ROOT / "data" / "processed" / "segment_tables.csv"
DEFAULT_MARKDOWN_PATH = REPO_ROOT / "docs" / "segment_tables.md"


def price_bands(prices: Any, n_bands: int = 4) -> tuple[pd.Series, list[float]]:
    """Bandas por cuartiles del precio observado. Son descriptivas, no una regla de modelado."""

    # Conserva el índice: el resultado se asigna de vuelta a un marco que puede estar filtrado.
    values = pd.Series(prices).astype(float)
    edges = np.unique(np.quantile(values.to_numpy(), np.linspace(0, 1, n_bands + 1)))
    if len(edges) < 3:
        raise ValueError("No hay variación de precio suficiente para formar bandas.")
    labels = [f"Q{index + 1}" for index in range(len(edges) - 1)]
    bands = pd.cut(values, bins=edges, labels=labels, include_lowest=True)
    return bands.astype(str), [float(edge) for edge in edges]


def segment_metrics(
    frame: pd.DataFrame,
    by: str,
    min_segment_size: int = MIN_SEGMENT_SIZE,
) -> pd.DataFrame:
    """Métricas por valor de un segmento, con el tamaño de muestra de cada fila."""

    required = {"price_mxn", by, *QUANTILE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas para la tabla por segmento: {sorted(missing)}.")

    records: list[dict[str, Any]] = []
    for value, chunk in frame.groupby(by, dropna=False, observed=True):
        results = evaluate_quantile_predictions(chunk["price_mxn"], chunk[QUANTILE_COLUMNS])
        records.append(
            {
                "segment": by,
                "value": str(value),
                "n": int(len(chunk)),
                "median_price_mxn": float(chunk["price_mxn"].median()),
                **{key: results[key] for key in REPORTED_METRICS},
            }
        )

    table = pd.DataFrame.from_records(records).sort_values("n", ascending=False)
    # Marca explícita para que la interpretación no saque conclusiones de grupos diminutos.
    table["enough_evidence"] = table["n"] >= min_segment_size
    return table.reset_index(drop=True)


def build_segment_tables(
    frame: pd.DataFrame,
    segments: tuple[str, ...] = SEGMENT_COLUMNS,
    min_segment_size: int = MIN_SEGMENT_SIZE,
) -> pd.DataFrame:
    """Apila las tablas de todos los segmentos pedidos."""

    working = frame.copy()
    if "price_band" in segments and "price_band" not in working.columns:
        working["price_band"], _ = price_bands(working["price_mxn"])
    tables = [segment_metrics(working, by, min_segment_size) for by in segments]
    return pd.concat(tables, ignore_index=True)


def tables_by_model(
    frame: pd.DataFrame,
    segments: tuple[str, ...] = SEGMENT_COLUMNS,
    min_segment_size: int = MIN_SEGMENT_SIZE,
) -> pd.DataFrame:
    """Igual que build_segment_tables, separando por modelo cuando la columna existe."""

    if "model" not in frame.columns:
        return build_segment_tables(frame, segments, min_segment_size)
    parts = []
    for model, chunk in frame.groupby("model", observed=True):
        table = build_segment_tables(chunk, segments, min_segment_size)
        table.insert(0, "model", model)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def render_markdown(table: pd.DataFrame, edges: list[float] | None = None) -> str:
    """Documento legible para la interpretación, sin conclusiones escritas aquí."""

    lines = [
        "# Tablas por segmento",
        "",
        "Generado por `src/segments.py`. Los números son métricas de validación sobre junio, "
        "calculadas con `src/metrics.py` contra el precio original en MXN.",
        "",
        f"`enough_evidence` es falso cuando el segmento tiene menos de {MIN_SEGMENT_SIZE} "
        "observaciones: ahí cualquier conclusión descansa en muy poca evidencia.",
        "",
    ]
    if edges:
        formatted = ", ".join(f"{edge:,.0f}" for edge in edges)
        lines += [f"Bordes de las bandas de precio en MXN: {formatted}.", ""]

    for segment in table["segment"].unique():
        chunk = table[table["segment"] == segment]
        lines += [f"## {segment}", ""]
        columns = [column for column in chunk.columns if column != "segment"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join(["---"] * len(columns)) + "|")
        for _, row in chunk.iterrows():
            cells = [
                f"{row[column]:,.4f}" if isinstance(row[column], float) else str(row[column])
                for column in columns
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def load_prediction_frame(
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    june_path: Path | None = None,
) -> pd.DataFrame:
    """Une las predicciones guardadas con las columnas de segmento del snapshot de junio."""

    predictions = pd.read_csv(predictions_path, dtype={"id": "string", "host_id": "string"})
    X, _, metadata = load_development() if june_path is None else load_development(june_path)
    segments = metadata.assign(
        neighbourhood_cleansed=X["neighbourhood_cleansed"].to_numpy(),
        room_type=X["room_type"].to_numpy(),
    )[["id", "neighbourhood_cleansed", "room_type"]]
    return predictions.merge(segments, on="id", how="left", validate="many_to_one")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES_PATH)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--model", type=str, default=None, help="Limitar a un modelo.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_prediction_frame(args.predictions)
    if args.model:
        frame = frame[frame["model"] == args.model]
    frame["price_band"], edges = price_bands(frame["price_mxn"])

    table = tables_by_model(frame)
    args.tables.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.tables, index=False)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(table, edges), encoding="utf-8")

    weak = int((~table["enough_evidence"]).sum())
    print(f"Segmentos: {len(table)} filas | sin evidencia suficiente: {weak}")
    print(f"Tablas: {args.tables}")
    print(f"Documento: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
