"""Pruebas de las tablas por segmento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.segments import (
    MIN_SEGMENT_SIZE,
    build_segment_tables,
    price_bands,
    render_markdown,
    segment_metrics,
    tables_by_model,
)


def prediction_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = pd.Series(rng.uniform(400, 4000, n))
    return pd.DataFrame(
        {
            "price_mxn": price,
            "neighbourhood_cleansed": rng.choice(["Centro", "Sur", "Chica"], n, p=[0.5, 0.45, 0.05]),
            "room_type": rng.choice(["Entire home/apt", "Private room"], n),
            "p10": price * 0.7,
            "p35": price * 0.9,
            "p50": price * 1.0,
            "p90": price * 1.3,
        }
    )


def test_price_bands_split_into_quartiles_with_edges() -> None:
    bands, edges = price_bands([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
    assert sorted(set(bands)) == ["Q1", "Q2", "Q3", "Q4"]
    assert len(edges) == 5
    assert edges == sorted(edges)


def test_price_bands_reject_a_constant_price() -> None:
    with pytest.raises(ValueError):
        price_bands([500.0] * 10)


def test_segment_metrics_report_sample_sizes_and_flag_weak_segments() -> None:
    table = segment_metrics(prediction_frame(), "neighbourhood_cleansed")

    assert set(table.columns) >= {"segment", "value", "n", "coverage", "enough_evidence"}
    assert table["n"].sum() == 200
    # El segmento chico queda marcado para que no se saquen conclusiones de él.
    small = table[table["n"] < MIN_SEGMENT_SIZE]
    assert (~small["enough_evidence"]).all()
    assert table["n"].is_monotonic_decreasing


def test_segment_metrics_require_the_prediction_columns() -> None:
    frame = prediction_frame().drop(columns=["p35"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        segment_metrics(frame, "room_type")


def test_build_segment_tables_covers_every_requested_segment() -> None:
    table = build_segment_tables(prediction_frame())
    assert set(table["segment"]) == {"neighbourhood_cleansed", "room_type", "price_band"}
    # Cada segmento reparte todas las filas evaluadas.
    for segment, chunk in table.groupby("segment"):
        assert chunk["n"].sum() == 200


def test_tables_by_model_separates_each_model() -> None:
    frame = pd.concat(
        [prediction_frame().assign(model="baseline"), prediction_frame().assign(model="lightgbm")],
        ignore_index=True,
    )
    table = tables_by_model(frame)
    assert set(table["model"]) == {"baseline", "lightgbm"}


def test_render_markdown_lists_every_segment_and_the_band_edges() -> None:
    table = build_segment_tables(prediction_frame())
    text = render_markdown(table, edges=[100.0, 500.0, 900.0])

    assert "## neighbourhood_cleansed" in text
    assert "## price_band" in text
    assert "enough_evidence" in text
    assert "Bordes de las bandas" in text


def test_price_bands_align_with_a_filtered_frame() -> None:
    frame = pd.concat(
        [prediction_frame().assign(model="baseline"), prediction_frame().assign(model="lightgbm")],
        ignore_index=True,
    )
    subset = frame[frame["model"] == "lightgbm"]
    bands, _ = price_bands(subset["price_mxn"])

    # El índice del subconjunto no empieza en cero; las bandas deben seguir alineando.
    assert subset.index.min() > 0
    subset = subset.assign(price_band=bands)
    assert subset["price_band"].notna().all()
    assert sorted(subset["price_band"].unique()) == ["Q1", "Q2", "Q3", "Q4"]
