"""Métricas comunes en MXN para comparar baseline, Ridge y LightGBM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


# Contrato de cuantiles acordado antes de comparar modelos (ver docs/metrics_contract.md).
LOWER_QUANTILE = 0.10
RECOMMENDED_QUANTILE = 0.35
MEDIAN_QUANTILE = 0.50
UPPER_QUANTILE = 0.90

LOWER_COLUMN = "p10"
RECOMMENDED_COLUMN = "p35"
MEDIAN_COLUMN = "p50"
UPPER_COLUMN = "p90"

QUANTILE_LEVELS: dict[str, float] = {
    LOWER_COLUMN: LOWER_QUANTILE,
    RECOMMENDED_COLUMN: RECOMMENDED_QUANTILE,
    MEDIAN_COLUMN: MEDIAN_QUANTILE,
    UPPER_COLUMN: UPPER_QUANTILE,
}

# Cobertura que el intervalo p10-p90 debería alcanzar si los cuantiles estuvieran bien calibrados.
NOMINAL_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE

SELECTION_METRIC = "pinball_recommended"


def _as_float_array(values: Any, name: str) -> np.ndarray:
    """Convierte a arreglo 1-D de float y rechaza valores no finitos."""

    array = np.asarray(pd.Series(values).to_numpy(), dtype="float64")
    if array.ndim != 1:
        raise ValueError(f"{name} debe ser unidimensional; se recibió forma {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} está vacío; no hay observaciones para evaluar.")
    if not np.isfinite(array).all():
        # No se descartan filas difíciles: una métrica sobre un subconjunto silencioso no es comparable.
        raise ValueError(f"{name} contiene valores nulos o no finitos; corregir antes de evaluar.")
    return array


def _check_same_length(reference: np.ndarray, other: np.ndarray, name: str) -> None:
    if reference.shape[0] != other.shape[0]:
        raise ValueError(
            f"{name} tiene {other.shape[0]} filas y el objetivo tiene {reference.shape[0]}."
        )


def validate_price_target(y_true: Any) -> np.ndarray:
    """Valida un objetivo de precio en MXN original y lo devuelve como arreglo."""

    array = _as_float_array(y_true, "y_true")
    if (array <= 0).any():
        # Guarda contra evaluar sobre log(precio) o sobre un objetivo recortado en vez de MXN.
        raise ValueError("y_true debe estar en MXN original y ser estrictamente positivo.")
    return array


def pinball_loss(y_true: Any, y_pred: Any, quantile: float) -> float:
    """Pérdida pinball promedio para un cuantil."""

    if not 0.0 < quantile < 1.0:
        raise ValueError(f"El cuantil debe estar en (0, 1); se recibió {quantile}.")
    target = validate_price_target(y_true)
    prediction = _as_float_array(y_pred, "y_pred")
    _check_same_length(target, prediction, "y_pred")
    error = target - prediction
    # Penaliza de forma asimétrica: subestimar cuesta q, sobreestimar cuesta 1-q.
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """MAE en MXN."""

    target = validate_price_target(y_true)
    prediction = _as_float_array(y_pred, "y_pred")
    _check_same_length(target, prediction, "y_pred")
    return float(np.mean(np.abs(target - prediction)))


def interval_coverage(y_true: Any, lower: Any, upper: Any) -> float:
    """Proporción de precios reales dentro del intervalo, con extremos incluidos."""

    target = validate_price_target(y_true)
    low = _as_float_array(lower, "lower")
    high = _as_float_array(upper, "upper")
    _check_same_length(target, low, "lower")
    _check_same_length(target, high, "upper")
    return float(np.mean((target >= low) & (target <= high)))


def mean_interval_width(lower: Any, upper: Any) -> float:
    """Amplitud media del intervalo en MXN."""

    low = _as_float_array(lower, "lower")
    high = _as_float_array(upper, "upper")
    _check_same_length(low, high, "upper")
    return float(np.mean(high - low))


def _ordered_prediction_frame(
    predictions: Any,
    quantile_levels: Mapping[str, float] = QUANTILE_LEVELS,
) -> pd.DataFrame:
    """Valida que estén las columnas del contrato y las ordena por nivel de cuantil."""

    frame = predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame(dict(predictions))
    missing = [column for column in quantile_levels if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas del contrato de predicción: {sorted(missing)}.")
    ordered = sorted(quantile_levels, key=lambda column: quantile_levels[column])
    return frame.loc[:, ordered]


def quantile_crossing_rate(
    predictions: Any,
    quantile_levels: Mapping[str, float] = QUANTILE_LEVELS,
) -> float:
    """Proporción de filas donde los cuantiles predichos no son monótonos."""

    ordered = _ordered_prediction_frame(predictions, quantile_levels)
    values = np.column_stack(
        [_as_float_array(ordered[column], column) for column in ordered.columns]
    )
    crossing = (np.diff(values, axis=1) < 0).any(axis=1)
    return float(np.mean(crossing))


def evaluate_quantile_predictions(
    y_true: Any,
    predictions: Any,
    quantile_levels: Mapping[str, float] = QUANTILE_LEVELS,
) -> dict[str, float]:
    """Calcula todas las métricas de una partición contra el precio original en MXN."""

    target = validate_price_target(y_true)
    ordered = _ordered_prediction_frame(predictions, quantile_levels)
    for column in ordered.columns:
        _check_same_length(target, _as_float_array(ordered[column], column), column)

    results: dict[str, float] = {"n_observations": float(target.shape[0])}
    for column, quantile in quantile_levels.items():
        results[f"pinball_{column}"] = pinball_loss(target, ordered[column], quantile)

    # Criterio de selección declarado: pinball del cuantil recomendado, no del promedio de cuantiles.
    results[SELECTION_METRIC] = results[f"pinball_{RECOMMENDED_COLUMN}"]
    results["mae_recommended"] = mean_absolute_error(target, ordered[RECOMMENDED_COLUMN])
    results["mae_median"] = mean_absolute_error(target, ordered[MEDIAN_COLUMN])
    results["coverage"] = interval_coverage(target, ordered[LOWER_COLUMN], ordered[UPPER_COLUMN])
    results["nominal_coverage"] = float(NOMINAL_COVERAGE)
    results["coverage_gap"] = results["coverage"] - float(NOMINAL_COVERAGE)
    results["mean_interval_width"] = mean_interval_width(
        ordered[LOWER_COLUMN], ordered[UPPER_COLUMN]
    )
    results["quantile_crossing_rate"] = quantile_crossing_rate(ordered, quantile_levels)
    return results


def summarize_fold_metrics(fold_metrics: Sequence[Mapping[str, float]]) -> pd.DataFrame:
    """Resume métricas por partición en media, desviación y rango."""

    if not fold_metrics:
        raise ValueError("No hay métricas por partición para resumir.")
    frame = pd.DataFrame(list(fold_metrics))
    summary = frame.agg(["mean", "std", "min", "max"]).transpose()
    summary["n_folds"] = len(frame)
    return summary
