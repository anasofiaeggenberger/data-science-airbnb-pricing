"""LightGBM cuantílico: un modelo por cuantil sobre los predictores y folds comunes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.metrics import QUANTILE_LEVELS, quantile_crossing_rate, validate_price_target
from src.modeling_data import feature_columns, prepare_features


MISSING_LABEL = "Missing"
OTHER_LABEL = "Other"
MIN_CATEGORY_COUNT = 20

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
}

# Búsqueda acotada: 12 configuraciones, ejecutadas y registradas por el ejecutor de experimentos.
PARAM_GRID: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63],
    "learning_rate": [0.05, 0.10],
    "min_child_samples": [20, 50],
}


@dataclass
class LightGBMQuantileModel:
    """Cuatro modelos LightGBM, uno por cuantil, bajo el contrato común.

    Reglas documentadas en docs/lightgbm_rules.md. Usa las columnas de
    `feature_columns` para compartir predictores con Ridge y el baseline.
    """

    include_optional: bool = False
    min_category_count: int = MIN_CATEGORY_COUNT
    quantile_levels: Mapping[str, float] = field(
        default_factory=lambda: dict(QUANTILE_LEVELS)
    )
    params: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        if self.min_category_count < 1:
            raise ValueError("min_category_count debe ser al menos 1.")
        self.models_: dict[str, LGBMRegressor] = {}
        self.categories_: dict[str, list[str]] = {}
        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.n_training_rows_: int = 0
        self.fitted_: bool = False

    @property
    def quantile_columns(self) -> list[str]:
        """Columnas del contrato ordenadas por nivel de cuantil."""

        return sorted(self.quantile_levels, key=lambda column: self.quantile_levels[column])

    def _normalized_labels(self, values: pd.Series) -> pd.Series:
        text = values.astype("string").str.strip()
        return text.mask(text.isna() | (text == ""), MISSING_LABEL)

    def _transform(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Selecciona predictores y codifica categorías como tipo category de pandas."""

        frame = prepare_features(X).loc[:, self.numeric_features_ + self.categorical_features_]
        for column in self.numeric_features_:
            # LightGBM maneja faltantes de forma nativa; no se imputa nada aquí.
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")

        for column in self.categorical_features_:
            labels = self._normalized_labels(frame[column])
            if fit:
                counts = labels.value_counts()
                kept = sorted(counts[counts >= self.min_category_count].index.astype(str))
                self.categories_[column] = kept + [OTHER_LABEL]
            known = set(self.categories_[column]) - {OTHER_LABEL}
            # Categorías raras en entrenamiento y desconocidas en predicción caen en la misma etiqueta.
            labels = labels.where(labels.isin(known), OTHER_LABEL)
            frame[column] = pd.Categorical(labels, categories=self.categories_[column])
        return frame

    def fit(self, X: pd.DataFrame, y: Any, groups: Any = None) -> "LightGBMQuantileModel":
        """Entrena un modelo por cuantil. `groups` se acepta y se ignora: no hay calibración interna."""

        target = validate_price_target(y)
        numeric, categorical = feature_columns(self.include_optional)
        self.numeric_features_ = list(numeric)
        self.categorical_features_ = list(categorical)

        frame = self._transform(X, fit=True)
        if len(frame) != target.shape[0]:
            raise ValueError(f"X tiene {len(frame)} filas y el objetivo {target.shape[0]}.")

        self.models_ = {}
        for column in self.quantile_columns:
            model = LGBMRegressor(
                objective="quantile",
                alpha=self.quantile_levels[column],
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=-1,
                **dict(self.params),
            )
            model.fit(frame, target, categorical_feature=self.categorical_features_)
            self.models_[column] = model

        self.n_training_rows_ = int(target.shape[0])
        self.fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("El modelo no ha sido ajustado; llamar fit antes de predict.")

    def predict_raw(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicciones tal como salen de los cuatro modelos, sin corregir cruces."""

        self._check_fitted()
        frame = self._transform(X)
        predictions = {
            column: self.models_[column].predict(frame) for column in self.quantile_columns
        }
        return pd.DataFrame(predictions, index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Devuelve p10, p35, p50 y p90 en MXN, con los cruces ya resueltos."""

        raw = self.predict_raw(X)
        # Los cuantiles se entrenan por separado y pueden cruzarse: se reordena cada fila.
        ordered = np.sort(raw.to_numpy(), axis=1)
        return pd.DataFrame(ordered, columns=self.quantile_columns, index=X.index)

    def crossing_rate(self, X: pd.DataFrame) -> float:
        """Proporción de filas que necesitaron corrección de cruces."""

        return quantile_crossing_rate(self.predict_raw(X), self.quantile_levels)

    def feature_importances(self) -> pd.DataFrame:
        """Importancia por variable y cuantil, para la evidencia de interpretación."""

        self._check_fitted()
        columns = self.numeric_features_ + self.categorical_features_
        return pd.DataFrame(
            {column: self.models_[column].feature_importances_ for column in self.quantile_columns},
            index=columns,
        )
