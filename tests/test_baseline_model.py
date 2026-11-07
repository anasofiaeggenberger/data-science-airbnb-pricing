"""Pruebas del baseline con grupos pequeños, categorías desconocidas y valores a mano."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baseline_model import (
    GROUP_COLUMNS,
    LEVEL_FALLBACK,
    LEVEL_GLOBAL,
    LEVEL_GROUP,
    MIN_GROUP_SIZE,
    BaselineQuantileModel,
)
from src.metrics import evaluate_quantile_predictions, quantile_crossing_rate


# Entrenamiento pequeño y verificable: un grupo suficiente, dos insuficientes.
TRAIN_X = pd.DataFrame(
    {
        "neighbourhood_cleansed": [
            "Benito Juárez",
            "Benito Juárez",
            "Benito Juárez",
            "Benito Juárez",
            "Benito Juárez",
            "Coyoacán",
            "Coyoacán",
            "Coyoacán",
        ],
        "room_type": [
            "Entire home/apt",
            "Entire home/apt",
            "Entire home/apt",
            "Entire home/apt",
            "Private room",
            "Private room",
            "Private room",
            "Private room",
        ],
    }
)
TRAIN_Y = [100.0, 200.0, 300.0, 400.0, 50.0, 60.0, 70.0, 80.0]


def fitted_model() -> BaselineQuantileModel:
    return BaselineQuantileModel(min_group_size=4).fit(TRAIN_X, TRAIN_Y)


def test_defaults_follow_the_proposal() -> None:
    model = BaselineQuantileModel()
    assert model.group_columns == GROUP_COLUMNS == ("neighbourhood_cleansed", "room_type")
    assert model.min_group_size == MIN_GROUP_SIZE == 20
    assert model.quantile_columns == ["p10", "p35", "p50", "p90"]


def test_large_group_uses_its_own_empirical_quantiles() -> None:
    model = fitted_model()
    query = pd.DataFrame(
        {"neighbourhood_cleansed": ["Benito Juárez"], "room_type": ["Entire home/apt"]}
    )
    prediction = model.predict_with_source(query).iloc[0]

    assert prediction["group_level"] == LEVEL_GROUP
    assert prediction["group_size"] == 4
    # Cuantiles de [100, 200, 300, 400] con interpolación lineal.
    assert prediction["p10"] == pytest.approx(130.0)
    assert prediction["p35"] == pytest.approx(205.0)
    assert prediction["p50"] == pytest.approx(250.0)
    assert prediction["p90"] == pytest.approx(370.0)
    # La mediana del grupo es la exigida por la propuesta original.
    assert prediction["p50"] == pytest.approx(np.median([100.0, 200.0, 300.0, 400.0]))


def test_small_group_falls_back_to_room_type() -> None:
    model = fitted_model()
    query = pd.DataFrame(
        {"neighbourhood_cleansed": ["Benito Juárez"], "room_type": ["Private room"]}
    )
    prediction = model.predict_with_source(query).iloc[0]

    assert prediction["group_level"] == LEVEL_FALLBACK
    assert prediction["group_size"] == 4
    # Cuantiles de los cuatro Private room: [50, 60, 70, 80].
    assert prediction["p50"] == pytest.approx(65.0)
    assert prediction["p10"] == pytest.approx(53.0)
    assert prediction["p90"] == pytest.approx(77.0)


def test_unknown_neighbourhood_still_uses_the_room_type_fallback() -> None:
    model = fitted_model()
    query = pd.DataFrame(
        {"neighbourhood_cleansed": ["Alcaldía Inexistente"], "room_type": ["Private room"]}
    )
    prediction = model.predict_with_source(query).iloc[0]

    assert prediction["group_level"] == LEVEL_FALLBACK
    assert prediction["p50"] == pytest.approx(65.0)


def test_unknown_room_type_falls_back_to_the_global_distribution() -> None:
    model = fitted_model()
    query = pd.DataFrame(
        {"neighbourhood_cleansed": ["Coyoacán"], "room_type": ["Hotel room"]}
    )
    prediction = model.predict_with_source(query).iloc[0]

    assert prediction["group_level"] == LEVEL_GLOBAL
    assert prediction["group_size"] == 8
    # Cuantiles de los ocho precios de entrenamiento.
    assert prediction["p50"] == pytest.approx(90.0)
    assert prediction["p10"] == pytest.approx(57.0)
    assert prediction["p90"] == pytest.approx(330.0)


def test_missing_categories_are_labeled_and_still_predict() -> None:
    model = fitted_model()
    query = pd.DataFrame(
        {"neighbourhood_cleansed": [None, ""], "room_type": ["Private room", np.nan]}
    )
    prediction = model.predict_with_source(query)

    assert prediction["group_level"].tolist() == [LEVEL_FALLBACK, LEVEL_GLOBAL]
    assert prediction[["p10", "p35", "p50", "p90"]].notna().all().all()


def test_predict_respects_the_common_contract() -> None:
    model = fitted_model()
    query = TRAIN_X.set_index(pd.Index([f"listing-{i}" for i in range(len(TRAIN_X))]))
    prediction = model.predict(query)

    assert list(prediction.columns) == ["p10", "p35", "p50", "p90"]
    assert prediction.index.equals(query.index)
    # Los cuantiles empíricos de una misma muestra no se cruzan.
    assert quantile_crossing_rate(prediction) == pytest.approx(0.0)


def test_rules_are_fitted_only_on_the_rows_given_to_fit() -> None:
    training = TRAIN_X.iloc[:4]
    model = BaselineQuantileModel(min_group_size=4).fit(training, TRAIN_Y[:4])

    assert model.global_size_ == 4
    assert model.global_quantiles_["p50"] == pytest.approx(250.0)
    # Los precios que no entraron a fit no crean reglas de respaldo.
    assert set(model.fallback_quantiles_) == {"Entire home/apt"}


def test_metrics_can_consume_baseline_predictions() -> None:
    model = fitted_model()
    validation_y = [150.0, 70.0]
    validation_X = pd.DataFrame(
        {
            "neighbourhood_cleansed": ["Benito Juárez", "Coyoacán"],
            "room_type": ["Entire home/apt", "Private room"],
        }
    )
    results = evaluate_quantile_predictions(validation_y, model.predict(validation_X))

    assert results["n_observations"] == 2
    assert results["quantile_crossing_rate"] == pytest.approx(0.0)
    assert results["mean_interval_width"] > 0


def test_invalid_usage_is_rejected() -> None:
    with pytest.raises(ValueError):
        BaselineQuantileModel(min_group_size=0)
    with pytest.raises(ValueError):
        BaselineQuantileModel(fallback_column="property_type")
    with pytest.raises(RuntimeError):
        BaselineQuantileModel().predict(TRAIN_X)
    with pytest.raises(ValueError, match="agrupación"):
        BaselineQuantileModel().fit(TRAIN_X.drop(columns=["room_type"]), TRAIN_Y)
    with pytest.raises(ValueError):
        BaselineQuantileModel(min_group_size=4).fit(TRAIN_X, TRAIN_Y[:4])
    with pytest.raises(ValueError, match="MXN"):
        BaselineQuantileModel(min_group_size=4).fit(TRAIN_X, [0.0] + TRAIN_Y[1:])
