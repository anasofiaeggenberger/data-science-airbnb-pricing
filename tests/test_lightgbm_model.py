"""Pruebas de LightGBM cuantílico: contrato, categorías nuevas y cruces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.lightgbm_model import OTHER_LABEL, PARAM_GRID, LightGBMQuantileModel
from src.metrics import evaluate_quantile_predictions, quantile_crossing_rate

# Modelos pequeños: las pruebas verifican comportamiento, no calidad predictiva.
FAST_PARAMS = {
    "n_estimators": 40,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 10,
}


def synthetic_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
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
            "neighbourhood_cleansed": rng.choice(["Centro", "Sur", "Rara"], n, p=[0.5, 0.47, 0.03]),
            "property_type": rng.choice(["Apartment", "House"], n),
            "room_type": rng.choice(["Entire home/apt", "Private room"], n),
            "host_has_profile_pic": ["t"] * n,
            "host_identity_verified": ["f"] * n,
        }
    )
    y = pd.Series(600 + X["accommodates"] * 320 + rng.normal(0, 120, n)).abs() + 50
    return X, y


def fitted_model(**kwargs) -> tuple[LightGBMQuantileModel, pd.DataFrame, pd.Series]:
    X, y = synthetic_frame()
    model = LightGBMQuantileModel(params=FAST_PARAMS, **kwargs).fit(X, y)
    return model, X, y


def test_predict_respects_the_common_contract() -> None:
    model, X, _ = fitted_model()
    query = X.set_index(pd.Index([f"listing-{i}" for i in range(len(X))]))
    prediction = model.predict(query)

    assert list(prediction.columns) == ["p10", "p35", "p50", "p90"]
    assert prediction.index.equals(query.index)
    assert prediction.notna().all().all()


def test_independent_quantiles_can_cross_but_predict_never_does() -> None:
    model, X, _ = fitted_model()
    # Entrenar cada cuantil por separado permite cruces; el contrato los resuelve.
    assert 0.0 <= model.crossing_rate(X) <= 1.0
    assert quantile_crossing_rate(model.predict(X)) == pytest.approx(0.0)


def test_rare_training_categories_are_grouped() -> None:
    model, _, _ = fitted_model()
    categories = model.categories_["neighbourhood_cleansed"]

    assert OTHER_LABEL in categories
    # "Rara" aparece en cerca del 3 % de las filas y no alcanza el mínimo.
    assert "Rara" not in categories
    assert {"Centro", "Sur"}.issubset(set(categories))


def test_unknown_categories_at_predict_time_resolve_to_other() -> None:
    model, X, _ = fitted_model()
    query = X.head(5).copy()
    query["neighbourhood_cleansed"] = "Alcaldía Inexistente"
    query["property_type"] = "Castillo"

    prediction = model.predict(query)
    assert prediction.notna().all().all()
    assert len(prediction) == 5


def test_missing_values_are_left_to_lightgbm() -> None:
    model, X, _ = fitted_model()
    query = X.head(5).copy()
    query.loc[query.index[0], "bedrooms"] = np.nan
    query.loc[query.index[1], "room_type"] = None

    prediction = model.predict(query)
    assert prediction.notna().all().all()


def test_optional_features_change_the_fitted_predictors() -> None:
    without, _, _ = fitted_model(include_optional=False)
    with_optional, _, _ = fitted_model(include_optional=True)

    assert "minimum_nights" not in without.numeric_features_
    assert "minimum_nights" in with_optional.numeric_features_
    assert "host_identity_verified" in with_optional.categorical_features_


def test_same_seed_gives_the_same_predictions() -> None:
    X, y = synthetic_frame()
    first = LightGBMQuantileModel(params=FAST_PARAMS).fit(X, y).predict(X)
    second = LightGBMQuantileModel(params=FAST_PARAMS).fit(X, y).predict(X)
    pd.testing.assert_frame_equal(first, second)


def test_metrics_can_consume_lightgbm_predictions() -> None:
    model, X, y = fitted_model()
    results = evaluate_quantile_predictions(y, model.predict(X))

    assert results["n_observations"] == len(X)
    assert results["quantile_crossing_rate"] == pytest.approx(0.0)
    assert results["mean_interval_width"] > 0


def test_feature_importances_cover_every_predictor_and_quantile() -> None:
    model, _, _ = fitted_model()
    importances = model.feature_importances()

    assert list(importances.columns) == ["p10", "p35", "p50", "p90"]
    assert set(importances.index) == set(model.numeric_features_ + model.categorical_features_)


def test_invalid_usage_is_rejected() -> None:
    X, y = synthetic_frame(n=60)
    with pytest.raises(RuntimeError):
        LightGBMQuantileModel(params=FAST_PARAMS).predict(X)
    with pytest.raises(ValueError):
        LightGBMQuantileModel(min_category_count=0)
    with pytest.raises(ValueError, match="MXN"):
        LightGBMQuantileModel(params=FAST_PARAMS).fit(X, [0.0] + list(y[1:]))
    with pytest.raises(ValueError):
        LightGBMQuantileModel(params=FAST_PARAMS).fit(X, y[:10])


def test_param_grid_is_small_and_explicit() -> None:
    combinations = 1
    for values in PARAM_GRID.values():
        combinations *= len(values)
    assert combinations == 12
