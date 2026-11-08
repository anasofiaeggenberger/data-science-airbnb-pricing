"""Pruebas de métricas con ejemplos calculados a mano."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    NOMINAL_COVERAGE,
    QUANTILE_LEVELS,
    SELECTION_METRIC,
    evaluate_quantile_predictions,
    interval_coverage,
    mean_absolute_error,
    mean_interval_width,
    pinball_loss,
    quantile_crossing_rate,
    summarize_fold_metrics,
)


def test_quantile_contract_matches_agreed_configuration() -> None:
    assert QUANTILE_LEVELS == {"p10": 0.10, "p35": 0.35, "p50": 0.50, "p90": 0.90}
    assert NOMINAL_COVERAGE == pytest.approx(0.80)


def test_pinball_loss_penalizes_both_sides_with_known_values() -> None:
    # Subestimar 20 con q=0.35 cuesta 7; sobreestimar 20 cuesta 13.
    assert pinball_loss([100.0], [80.0], 0.35) == pytest.approx(7.0)
    assert pinball_loss([100.0], [120.0], 0.35) == pytest.approx(13.0)
    assert pinball_loss([100.0, 100.0], [80.0, 120.0], 0.35) == pytest.approx(10.0)


def test_pinball_loss_at_median_is_half_mae() -> None:
    y_true = [100.0, 200.0, 300.0]
    y_pred = [120.0, 150.0, 330.0]
    assert pinball_loss(y_true, y_pred, 0.50) == pytest.approx(
        mean_absolute_error(y_true, y_pred) / 2
    )


def test_pinball_loss_is_zero_for_exact_predictions() -> None:
    y_true = [100.0, 250.0]
    for quantile in QUANTILE_LEVELS.values():
        assert pinball_loss(y_true, y_true, quantile) == pytest.approx(0.0)


def test_pinball_loss_rejects_quantiles_outside_the_open_interval() -> None:
    for quantile in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            pinball_loss([100.0], [90.0], quantile)


def test_interval_coverage_counts_the_bounds_as_covered() -> None:
    y_true = [100.0, 200.0, 300.0]
    lower = [100.0, 150.0, 350.0]
    upper = [150.0, 250.0, 400.0]
    assert interval_coverage(y_true, lower, upper) == pytest.approx(2 / 3)
    assert mean_interval_width(lower, upper) == pytest.approx(200 / 3)


def test_quantile_crossing_rate_detects_non_monotonic_rows() -> None:
    predictions = pd.DataFrame(
        {
            "p10": [10.0, 10.0],
            "p35": [20.0, 20.0],
            "p50": [15.0, 30.0],  # la primera fila cruza: p50 por debajo de p35
            "p90": [40.0, 40.0],
        }
    )
    assert quantile_crossing_rate(predictions) == pytest.approx(0.5)


def test_evaluate_quantile_predictions_reports_every_required_metric() -> None:
    y_true = [100.0, 200.0]
    predictions = pd.DataFrame(
        {
            "p10": [50.0, 100.0],
            "p35": [80.0, 180.0],
            "p50": [90.0, 210.0],
            "p90": [150.0, 300.0],
        }
    )
    results = evaluate_quantile_predictions(y_true, predictions)

    assert results["n_observations"] == 2
    assert results["pinball_p10"] == pytest.approx(7.5)
    assert results["pinball_p35"] == pytest.approx(7.0)
    assert results["pinball_p50"] == pytest.approx(5.0)
    assert results["pinball_p90"] == pytest.approx(7.5)
    assert results[SELECTION_METRIC] == pytest.approx(results["pinball_p35"])
    # El MAE recomendado y el de la mediana se reportan por separado.
    assert results["mae_recommended"] == pytest.approx(20.0)
    assert results["mae_median"] == pytest.approx(10.0)
    assert results["coverage"] == pytest.approx(1.0)
    assert results["coverage_gap"] == pytest.approx(0.20)
    assert results["mean_interval_width"] == pytest.approx(150.0)
    assert results["quantile_crossing_rate"] == pytest.approx(0.0)


def test_evaluate_accepts_a_mapping_of_arrays() -> None:
    y_true = np.array([100.0, 200.0])
    predictions = {
        "p10": np.array([50.0, 100.0]),
        "p35": np.array([80.0, 180.0]),
        "p50": np.array([90.0, 210.0]),
        "p90": np.array([150.0, 300.0]),
    }
    assert evaluate_quantile_predictions(y_true, predictions)["mae_median"] == pytest.approx(10.0)


def test_evaluate_requires_the_four_contract_columns() -> None:
    y_true = [100.0]
    with pytest.raises(ValueError, match="contrato"):
        evaluate_quantile_predictions(y_true, {"p10": [50.0], "p50": [90.0], "p90": [150.0]})


def test_metrics_reject_missing_values_instead_of_dropping_rows() -> None:
    with pytest.raises(ValueError):
        mean_absolute_error([100.0, np.nan], [90.0, 90.0])
    with pytest.raises(ValueError):
        mean_absolute_error([100.0, 200.0], [90.0, np.nan])


def test_metrics_reject_targets_that_are_not_positive_mxn() -> None:
    # El objetivo de validación siempre es precio válido en MXN: cero o negativo es un error de uso.
    with pytest.raises(ValueError, match="MXN"):
        mean_absolute_error([0.0, 200.0], [90.0, 190.0])
    with pytest.raises(ValueError, match="MXN"):
        mean_absolute_error([100.0, -50.0], [90.0, 190.0])


def test_metrics_reject_length_mismatch_and_empty_input() -> None:
    with pytest.raises(ValueError):
        mean_absolute_error([100.0, 200.0], [90.0])
    with pytest.raises(ValueError):
        mean_absolute_error([], [])


def test_summarize_fold_metrics_averages_across_partitions() -> None:
    summary = summarize_fold_metrics(
        [
            {"pinball_recommended": 10.0, "coverage": 0.80},
            {"pinball_recommended": 20.0, "coverage": 0.90},
        ]
    )
    assert summary.loc["pinball_recommended", "mean"] == pytest.approx(15.0)
    assert summary.loc["pinball_recommended", "std"] == pytest.approx(7.0710678, rel=1e-6)
    assert summary.loc["coverage", "min"] == pytest.approx(0.80)
    assert summary.loc["coverage", "n_folds"] == 2
