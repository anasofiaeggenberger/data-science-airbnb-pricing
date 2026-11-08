"""Pruebas del ejecutor de experimentos: particiones compartidas, búsqueda y tope de precios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_pinball_loss

from src.baseline_model import BaselineQuantileModel
from src.lightgbm_model import LightGBMQuantileModel
from src.metrics import SELECTION_METRIC, pinball_loss
from src.ridge_model import RidgeQuantileModel
from src.train_models import (
    build_model,
    cross_validate,
    fold_assignment,
    fold_assignment_sha256,
    run_comparison,
    run_sensitivities,
    search_configuration,
    search_space,
)


def synthetic_development(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "latitude": rng.normal(19.4, 0.05, n),
            "longitude": rng.normal(-99.1, 0.05, n),
            "accommodates": rng.integers(1, 8, n),
            "bathrooms": rng.integers(1, 3, n),
            "bedrooms": rng.integers(1, 4, n),
            "beds": rng.integers(1, 5, n),
            "amenities": ['["Wifi", "TV"]'] * n,
            "minimum_nights": rng.integers(1, 5, n),
            "maximum_nights": rng.integers(30, 365, n),
            "neighbourhood_cleansed": rng.choice(["Centro", "Sur"], n),
            "property_type": rng.choice(["Apartment", "House"], n),
            "room_type": rng.choice(["Entire home/apt", "Private room"], n),
            "host_has_profile_pic": ["t"] * n,
            "host_identity_verified": ["f"] * n,
        }
    )
    y = pd.Series(600 + X["accommodates"] * 320 + rng.normal(0, 120, n)).abs() + 50
    metadata = pd.DataFrame(
        {"id": [f"L{i}" for i in range(n)], "host_id": [f"H{i % 40}" for i in range(n)]}
    )
    return X, y, metadata


def simple_folds(n: int, n_splits: int = 3) -> list[tuple]:
    positions = np.arange(n)
    return [
        (positions[positions % n_splits != fold], positions[positions % n_splits == fold])
        for fold in range(n_splits)
    ]


def test_search_space_sizes_match_the_documented_grids() -> None:
    assert len(search_space("baseline")) == 1
    assert len(search_space("ridge")) == 4
    assert len(search_space("lightgbm")) == 12
    assert len(search_space("lightgbm", quick=True)) == 1
    with pytest.raises(ValueError):
        search_space("red neuronal")


def test_build_model_returns_each_family_of_the_contract() -> None:
    assert isinstance(build_model("baseline", {"min_group_size": 20}), BaselineQuantileModel)
    assert isinstance(build_model("ridge", {"alpha": 1.0}), RidgeQuantileModel)
    assert isinstance(build_model("lightgbm", {"n_estimators": 10}), LightGBMQuantileModel)
    with pytest.raises(ValueError):
        build_model("red neuronal", {})


def test_our_pinball_loss_matches_the_scikit_learn_reference() -> None:
    rng = np.random.default_rng(3)
    y_true = np.abs(rng.normal(1500, 400, 200)) + 50
    y_pred = np.abs(rng.normal(1400, 400, 200)) + 50
    for quantile in (0.10, 0.35, 0.50, 0.90):
        assert pinball_loss(y_true, y_pred, quantile) == pytest.approx(
            mean_pinball_loss(y_true, y_pred, alpha=quantile)
        )


def test_fold_hash_is_deterministic_and_sensitive_to_the_partition() -> None:
    _, _, metadata = synthetic_development()
    folds = simple_folds(len(metadata))
    other = simple_folds(len(metadata), n_splits=5)

    assert fold_assignment_sha256(metadata, folds) == fold_assignment_sha256(metadata, folds)
    assert fold_assignment_sha256(metadata, folds) != fold_assignment_sha256(metadata, other)
    assignments = fold_assignment(metadata, folds)
    assert set(assignments.columns) == {"id", "host_id", "fold"}
    assert assignments["fold"].nunique() == 3


def test_cross_validate_covers_every_row_exactly_once() -> None:
    X, y, metadata = synthetic_development()
    folds = simple_folds(len(X))
    fold_metrics, predictions = cross_validate(
        "baseline", {"min_group_size": 10}, X, y, metadata, folds
    )

    assert len(fold_metrics) == 3
    assert predictions[["p10", "p35", "p50", "p90"]].notna().all().all()
    assert sum(int(record["n_observations"]) for record in fold_metrics) == len(X)
    assert sorted(predictions["fold"].unique()) == [0, 1, 2]


def test_price_cap_is_learned_inside_training_and_never_touches_validation() -> None:
    X, y, metadata = synthetic_development()
    y = y.copy()
    y.iloc[:10] = 90_000.0  # precios extremos
    original = y.copy()
    folds = simple_folds(len(X))

    plain, _ = cross_validate("baseline", {"min_group_size": 10}, X, y, metadata, folds)
    capped, _ = cross_validate(
        "baseline", {"min_group_size": 10}, X, y, metadata, folds, price_cap_quantile=0.99
    )

    # No se elimina ninguna fila de validación al aplicar el tope.
    assert [record["n_observations"] for record in plain] == [
        record["n_observations"] for record in capped
    ]
    # El objetivo original no se modifica en el camino.
    pd.testing.assert_series_equal(y, original)


def test_search_selects_the_configuration_with_the_lowest_pinball() -> None:
    X, y, metadata = synthetic_development()
    folds = simple_folds(len(X))
    result = search_configuration("ridge", X, y, metadata, folds)

    scores = [item["mean_selection_metric"] for item in result["evaluated"]]
    assert len(scores) == 4
    assert result["best"]["mean_selection_metric"] == pytest.approx(min(scores))


def test_run_comparison_uses_one_partition_set_and_records_everything() -> None:
    X, y, metadata = synthetic_development()
    folds = simple_folds(len(X))
    outcome = run_comparison(X, y, metadata, folds=folds, quick=True)
    record = outcome["record"]

    assert record["test_evaluated"] is False
    assert [item["model"] for item in record["models"]] == ["baseline", "ridge", "lightgbm"]
    assert record["winner"] in {"baseline", "ridge", "lightgbm"}
    assert record["fold_assignment_sha256"] == fold_assignment_sha256(metadata, folds)
    # El ganador es el de menor pinball del cuantil recomendado, no otro criterio.
    scores = {item["model"]: item["mean"][SELECTION_METRIC] for item in record["models"]}
    assert record["winner"] == min(scores, key=scores.get)
    # Las predicciones guardadas son trazables fila por fila y modelo por modelo.
    assert len(outcome["predictions"]) == 3 * len(X)
    assert {"id", "host_id", "fold", "model", "price_mxn"}.issubset(outcome["predictions"].columns)


def test_sensitivities_change_one_decision_at_a_time() -> None:
    X, y, metadata = synthetic_development()
    folds = simple_folds(len(X))
    selected = {"baseline": {"min_group_size": 10}, "ridge": {"alpha": 1.0}}

    results = run_sensitivities(
        X, y, metadata, selected, folds=folds, models=("baseline", "ridge")
    )
    records = results["results"]

    assert results["test_evaluated"] is False
    assert len(records) == 3 * 2
    assert {record["variant"] for record in records} == {
        "base",
        "optional_features_included",
        "extreme_prices_capped_p99",
    }
    assert all(record["status"] == "ok" for record in records)
    # La variante base es la referencia, así que no se mueve respecto de sí misma.
    for record in records:
        if record["variant"] == "base":
            assert record["change_vs_base_pct"] == pytest.approx(0.0)
    # El baseline no mira variables opcionales, así que esa variante no debería moverlo.
    baseline_optional = next(
        record
        for record in records
        if record["model"] == "baseline" and record["variant"] == "optional_features_included"
    )
    assert baseline_optional["change_vs_base_pct"] == pytest.approx(0.0)


def test_a_variant_that_breaks_a_model_is_recorded_instead_of_aborting(monkeypatch) -> None:
    from src import train_models

    X, y, metadata = synthetic_development()
    folds = simple_folds(len(X))
    real_cross_validate = train_models.cross_validate

    def failing(name, params, X_, y_, metadata_, folds_, include_optional=False, price_cap_quantile=None):
        if include_optional:
            raise ValueError("Predicción fuera del rango numérico; revisar las entradas.")
        return real_cross_validate(
            name, params, X_, y_, metadata_, folds_, include_optional, price_cap_quantile
        )

    monkeypatch.setattr(train_models, "cross_validate", failing)
    results = train_models.run_sensitivities(
        X, y, metadata, {"baseline": {"min_group_size": 10}}, folds=folds, models=("baseline",)
    )

    statuses = {record["variant"]: record["status"] for record in results["results"]}
    assert statuses["base"] == "ok"
    assert statuses["extreme_prices_capped_p99"] == "ok"
    # La variante rota queda registrada con su causa, y las demás siguen calculándose.
    assert statuses["optional_features_included"] == "failed"
    broken = next(
        record for record in results["results"] if record["status"] == "failed"
    )
    assert "rango numérico" in broken["error"]
    assert broken["mean"] is None
    assert broken["change_vs_base_pct"] is None


def test_crossing_rate_is_reported_before_and_after_sorting() -> None:
    X, y, metadata = synthetic_development()
    folds = simple_folds(len(X))

    lgbm_metrics, _ = cross_validate(
        "lightgbm",
        {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 10},
        X,
        y,
        metadata,
        folds,
    )
    for record in lgbm_metrics:
        # Después de ordenar nunca quedan cruces; antes es una cifra que hay que reportar aparte.
        assert record["quantile_crossing_rate"] == pytest.approx(0.0)
        assert 0.0 <= record["crossing_rate_before_sorting"] <= 1.0

    # Un modelo sin paso de ordenamiento reporta la misma cifra en ambas claves.
    baseline_metrics, _ = cross_validate(
        "baseline", {"min_group_size": 10}, X, y, metadata, folds
    )
    for record in baseline_metrics:
        assert record["crossing_rate_before_sorting"] == record["quantile_crossing_rate"]
