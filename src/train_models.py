"""Ejecutor de experimentos: compara baseline, Ridge y LightGBM sobre las mismas particiones."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baseline_model import MIN_GROUP_SIZE, BaselineQuantileModel
from src.lightgbm_model import DEFAULT_PARAMS, PARAM_GRID, LightGBMQuantileModel
from src.metrics import (
    QUANTILE_LEVELS,
    SELECTION_METRIC,
    evaluate_quantile_predictions,
    summarize_fold_metrics,
)
from src.modeling_data import load_development, make_folds
from src.preparation import DEFAULT_JUNE_PATH, REPO_ROOT
from src.ridge_model import RidgeQuantileModel


MODEL_NAMES = ("baseline", "ridge", "lightgbm")
QUANTILE_COLUMNS = sorted(QUANTILE_LEVELS, key=lambda column: QUANTILE_LEVELS[column])

DEFAULT_COMPARISON_PATH = REPO_ROOT / "docs" / "model_comparison.json"
DEFAULT_PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "validation_predictions.csv"
DEFAULT_SENSITIVITY_PATH = REPO_ROOT / "docs" / "sensitivity_results.json"

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)

# Alternativas acordadas: se cambia una cosa a la vez respecto de la especificación principal.
SENSITIVITIES = (
    {"name": "base", "include_optional": False, "price_cap_quantile": None},
    {"name": "optional_features_included", "include_optional": True, "price_cap_quantile": None},
    {"name": "extreme_prices_capped_p99", "include_optional": False, "price_cap_quantile": 0.99},
)


def search_space(name: str, quick: bool = False) -> list[dict[str, Any]]:
    """Configuraciones evaluadas por modelo. El baseline no se ajusta por validación."""

    if name == "baseline":
        return [{"min_group_size": MIN_GROUP_SIZE}]
    if name == "ridge":
        alphas = RIDGE_ALPHAS[:1] if quick else RIDGE_ALPHAS
        return [{"alpha": alpha} for alpha in alphas]
    if name == "lightgbm":
        if quick:
            return [dict(DEFAULT_PARAMS)]
        combinations = [dict(zip(PARAM_GRID, values)) for values in product(*PARAM_GRID.values())]
        return [{**DEFAULT_PARAMS, **combination} for combination in combinations]
    raise ValueError(f"Modelo desconocido: {name}.")


def build_model(name: str, params: dict[str, Any], include_optional: bool = False) -> Any:
    """Construye un modelo del contrato común a partir de su configuración."""

    if name == "baseline":
        # El baseline solo usa alcaldía y tipo; include_optional no lo afecta.
        return BaselineQuantileModel(**params)
    if name == "ridge":
        return RidgeQuantileModel(include_optional=include_optional, **params)
    if name == "lightgbm":
        return LightGBMQuantileModel(include_optional=include_optional, params=dict(params))
    raise ValueError(f"Modelo desconocido: {name}.")


def fold_assignment(metadata: pd.DataFrame, folds: list[tuple]) -> pd.DataFrame:
    """Tabla id/host_id/fold, idéntica a la que usa la validación de Ridge."""

    assignments = metadata.copy()
    for fold, (_, valid) in enumerate(folds):
        assignments.loc[valid, "fold"] = fold
    assignments["fold"] = assignments["fold"].astype(int)
    return assignments


def fold_assignment_sha256(metadata: pd.DataFrame, folds: list[tuple]) -> str:
    """Huella de las particiones, para probar que todos los modelos usaron las mismas."""

    return hashlib.sha256(fold_assignment(metadata, folds).to_csv(index=False).encode()).hexdigest()


def cross_validate(
    name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    folds: list[tuple],
    include_optional: bool = False,
    price_cap_quantile: float | None = None,
) -> tuple[list[dict[str, float]], pd.DataFrame]:
    """Evalúa una configuración en todas las particiones y devuelve predicciones fuera de fold."""

    fold_metrics: list[dict[str, float]] = []
    predictions = pd.DataFrame(index=X.index, columns=QUANTILE_COLUMNS, dtype=float)
    fold_ids = pd.Series(index=X.index, dtype="float64")

    for fold, (train, valid) in enumerate(folds):
        X_train, X_valid = X.iloc[train], X.iloc[valid]
        y_train, y_valid = y.iloc[train], y.iloc[valid]
        groups = metadata["host_id"].iloc[train]

        if price_cap_quantile is not None:
            # El tope se aprende dentro del entrenamiento; la validación conserva su precio original.
            cap = float(np.quantile(y_train.to_numpy(), price_cap_quantile))
            y_train = y_train.clip(upper=cap)

        model = build_model(name, params, include_optional).fit(X_train, y_train, groups=groups)
        fold_prediction = model.predict(X_valid)
        predictions.iloc[valid] = fold_prediction.to_numpy()
        fold_ids.iloc[valid] = fold

        results = evaluate_quantile_predictions(y_valid, fold_prediction)
        results["fold"] = float(fold)
        fold_metrics.append(results)

    predictions.insert(0, "fold", fold_ids.astype(int))
    return fold_metrics, predictions


def search_configuration(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    folds: list[tuple],
    include_optional: bool = False,
    quick: bool = False,
    price_cap_quantile: float | None = None,
) -> dict[str, Any]:
    """Recorre la búsqueda acotada y selecciona por pinball del cuantil recomendado."""

    evaluated: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for params in search_space(name, quick=quick):
        fold_metrics, predictions = cross_validate(
            name, params, X, y, metadata, folds, include_optional, price_cap_quantile
        )
        summary = summarize_fold_metrics(fold_metrics)
        score = float(summary.loc[SELECTION_METRIC, "mean"])
        evaluated.append(
            {
                "params": params,
                "mean_selection_metric": score,
                "std_selection_metric": float(summary.loc[SELECTION_METRIC, "std"]),
                "fold_metrics": fold_metrics,
            }
        )
        if best is None or score < best["mean_selection_metric"]:
            best = {
                "params": params,
                "mean_selection_metric": score,
                "fold_metrics": fold_metrics,
                "summary": summary,
                "predictions": predictions,
            }

    assert best is not None
    return {"model": name, "best": best, "evaluated": evaluated}


def _summary_record(name: str, result: dict[str, Any]) -> dict[str, Any]:
    summary = result["best"]["summary"]
    keys = [
        SELECTION_METRIC,
        "mae_recommended",
        "mae_median",
        "coverage",
        "mean_interval_width",
        "quantile_crossing_rate",
        "n_observations",
    ]
    return {
        "model": name,
        "selected_params": result["best"]["params"],
        "configurations_evaluated": len(result["evaluated"]),
        "mean": {key: float(summary.loc[key, "mean"]) for key in keys},
        "std": {key: float(summary.loc[key, "std"]) for key in keys},
        "fold_metrics": result["best"]["fold_metrics"],
        "search": [
            {"params": item["params"], "mean_selection_metric": item["mean_selection_metric"]}
            for item in result["evaluated"]
        ],
    }


def run_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    folds: list[tuple] | None = None,
    include_optional: bool = False,
    quick: bool = False,
    models: tuple[str, ...] = MODEL_NAMES,
) -> dict[str, Any]:
    """Compara todos los modelos sobre las mismas particiones y arma el registro completo."""

    folds = make_folds(metadata) if folds is None else folds
    results = {
        name: search_configuration(name, X, y, metadata, folds, include_optional, quick)
        for name in models
    }
    summaries = [_summary_record(name, results[name]) for name in models]
    winner = min(summaries, key=lambda record: record["mean"][SELECTION_METRIC])

    predictions = pd.concat(
        [
            metadata.assign(price_mxn=y.to_numpy(), model=name)
            .join(results[name]["best"]["predictions"])
            for name in models
        ],
        ignore_index=True,
    )

    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": "2026-06-15",
        "test_evaluated": False,
        "n_rows": int(len(X)),
        "n_hosts": int(metadata["host_id"].nunique()),
        "outer_folds": len(folds),
        "include_optional": include_optional,
        "quick": quick,
        "quantiles": QUANTILE_LEVELS,
        "selection_metric": SELECTION_METRIC,
        "fold_assignment_sha256": fold_assignment_sha256(metadata, folds),
        "june_sha256": hashlib.sha256(DEFAULT_JUNE_PATH.read_bytes()).hexdigest()
        if DEFAULT_JUNE_PATH.exists()
        else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "models": summaries,
        "winner": winner["model"],
        "limitation": (
            "Validación usada para seleccionar configuración; no es una estimación independiente. "
            "El test de septiembre no participa."
        ),
    }
    return {"record": record, "predictions": predictions, "results": results}


def run_sensitivities(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    selected: dict[str, dict[str, Any]],
    folds: list[tuple] | None = None,
    models: tuple[str, ...] = MODEL_NAMES,
    variants: tuple[dict[str, Any], ...] = SENSITIVITIES,
) -> dict[str, Any]:
    """Repite la comparación cambiando una decisión a la vez, con la configuración ya elegida."""

    folds = make_folds(metadata) if folds is None else folds
    records: list[dict[str, Any]] = []

    for variant in variants:
        for name in models:
            params = selected[name]
            entry: dict[str, Any] = {
                "variant": variant["name"],
                "model": name,
                "params": params,
                "include_optional": variant["include_optional"],
                "price_cap_quantile": variant["price_cap_quantile"],
            }
            try:
                fold_metrics, _ = cross_validate(
                    name,
                    params,
                    X,
                    y,
                    metadata,
                    folds,
                    include_optional=variant["include_optional"],
                    price_cap_quantile=variant["price_cap_quantile"],
                )
            except (ValueError, FloatingPointError) as error:
                # Una variante que rompe un modelo es un resultado, no una razón para abortar todo.
                entry.update({"status": "failed", "error": str(error), "mean": None})
                records.append(entry)
                continue

            summary = summarize_fold_metrics(fold_metrics)
            entry.update(
                {
                    "status": "ok",
                    "mean": {
                        key: float(summary.loc[key, "mean"])
                        for key in (SELECTION_METRIC, "mae_recommended", "coverage",
                                    "mean_interval_width")
                    },
                    "std_selection_metric": float(summary.loc[SELECTION_METRIC, "std"]),
                }
            )
            records.append(entry)

    base = {
        record["model"]: record["mean"][SELECTION_METRIC]
        for record in records
        if record["variant"] == "base" and record["status"] == "ok"
    }
    for record in records:
        # Cuánto mueve cada alternativa al criterio de selección, en porcentaje.
        reference = base.get(record["model"])
        if record["status"] != "ok" or reference is None:
            record["change_vs_base_pct"] = None
            continue
        record["change_vs_base_pct"] = float(
            (record["mean"][SELECTION_METRIC] / reference - 1) * 100
        )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": False,
        "selection_metric": SELECTION_METRIC,
        "fold_assignment_sha256": fold_assignment_sha256(metadata, folds),
        "variants": [variant["name"] for variant in variants],
        "results": records,
        "note": (
            "Cada umbral se aprende dentro del entrenamiento de cada partición. "
            "Las métricas se calculan contra el precio original de validación."
        ),
    }


def selected_params_from_comparison(comparison_path: Path = DEFAULT_COMPARISON_PATH) -> dict:
    """Lee la configuración ganadora de cada modelo de una comparación ya ejecutada."""

    comparison = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    return {record["model"]: record["selected_params"] for record in comparison["models"]}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_comparison(
    outcome: dict[str, Any],
    comparison_path: Path = DEFAULT_COMPARISON_PATH,
    predictions_path: Path | None = DEFAULT_PREDICTIONS_PATH,
) -> None:
    """Guarda el registro en docs y las predicciones de validación en datos derivados."""

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        json.dumps(_json_ready(outcome["record"]), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        outcome["predictions"].to_csv(predictions_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Una configuración por modelo.")
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--sensitivities", type=Path, default=DEFAULT_SENSITIVITY_PATH)
    parser.add_argument(
        "--mode",
        choices=("compare", "sensitivities", "both"),
        default="compare",
        help="sensitivities reutiliza la configuración de una comparación ya guardada.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    X, y, metadata = load_development()

    if args.mode in ("compare", "both"):
        outcome = run_comparison(
            X, y, metadata, include_optional=args.include_optional, quick=args.quick
        )
        save_comparison(outcome, args.comparison, args.predictions)
        table = pd.DataFrame(
            [
                {"model": record["model"], **record["mean"], "params": record["selected_params"]}
                for record in outcome["record"]["models"]
            ]
        )
        print(table.to_string(index=False))
        print(f"\nGanador por {SELECTION_METRIC}: {outcome['record']['winner']}")
        print(f"Huella de particiones: {outcome['record']['fold_assignment_sha256']}")

    if args.mode in ("sensitivities", "both"):
        selected = selected_params_from_comparison(args.comparison)
        results = run_sensitivities(X, y, metadata, selected)
        args.sensitivities.parent.mkdir(parents=True, exist_ok=True)
        args.sensitivities.write_text(
            json.dumps(_json_ready(results), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sensitivity_table = pd.DataFrame(
            [
                {
                    "variant": record["variant"],
                    "model": record["model"],
                    "status": record["status"],
                    SELECTION_METRIC: (record["mean"] or {}).get(SELECTION_METRIC),
                    "coverage": (record["mean"] or {}).get("coverage"),
                    "change_vs_base_pct": record["change_vs_base_pct"],
                }
                for record in results["results"]
            ]
        )
        print("\n" + sensitivity_table.to_string(index=False))
        print(f"\nSensibilidades en {args.sensitivities}")

    print("Septiembre no evaluado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
