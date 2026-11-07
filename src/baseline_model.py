"""Baseline de cuantiles empíricos por alcaldía y tipo de alojamiento."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import QUANTILE_LEVELS, validate_price_target


GROUP_COLUMNS = ("neighbourhood_cleansed", "room_type")
FALLBACK_COLUMN = "room_type"
MIN_GROUP_SIZE = 20
MISSING_LABEL = "Missing"

LEVEL_GROUP = "group"
LEVEL_FALLBACK = "fallback"
LEVEL_GLOBAL = "global"


def _normalize_labels(values: pd.Series) -> pd.Series:
    """Normaliza categorías a texto; nulos y vacíos quedan como una etiqueta explícita."""

    text = values.astype("string").str.strip()
    return text.mask(text.isna() | (text == ""), MISSING_LABEL)


@dataclass
class BaselineQuantileModel:
    """Mediana y cuantiles empíricos por grupo, con respaldos para grupos pequeños.

    Reglas documentadas en docs/baseline_rules.md. Todos los valores se calculan
    únicamente con las filas de entrenamiento que recibe `fit`.
    """

    group_columns: tuple[str, ...] = GROUP_COLUMNS
    fallback_column: str = FALLBACK_COLUMN
    min_group_size: int = MIN_GROUP_SIZE
    quantile_levels: Mapping[str, float] = field(
        default_factory=lambda: dict(QUANTILE_LEVELS)
    )

    def __post_init__(self) -> None:
        if self.min_group_size < 1:
            raise ValueError("min_group_size debe ser al menos 1.")
        if self.fallback_column not in self.group_columns:
            raise ValueError(
                f"fallback_column '{self.fallback_column}' debe ser una de {self.group_columns}."
            )
        self.group_quantiles_: dict[tuple[str, ...], dict[str, float]] = {}
        self.fallback_quantiles_: dict[str, dict[str, float]] = {}
        self.global_quantiles_: dict[str, float] = {}
        self.group_sizes_: dict[tuple[str, ...], int] = {}
        self.fallback_sizes_: dict[str, int] = {}
        self.global_size_: int = 0
        self.fitted_: bool = False

    @property
    def quantile_columns(self) -> list[str]:
        """Columnas del contrato ordenadas por nivel de cuantil."""

        return sorted(self.quantile_levels, key=lambda column: self.quantile_levels[column])

    def _empirical_quantiles(self, values: np.ndarray) -> dict[str, float]:
        # Los cuantiles empíricos de una misma muestra nunca se cruzan.
        return {
            column: float(np.quantile(values, self.quantile_levels[column]))
            for column in self.quantile_columns
        }

    def _label_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.group_columns if column not in X.columns]
        if missing:
            raise ValueError(f"Faltan columnas de agrupación: {missing}.")
        return pd.DataFrame(
            {column: _normalize_labels(X[column]) for column in self.group_columns},
            index=X.index,
        )

    def fit(self, X: pd.DataFrame, y: Any) -> "BaselineQuantileModel":
        """Ajusta los cuantiles por grupo usando solo estas filas de entrenamiento."""

        target = validate_price_target(y)
        labels = self._label_frame(X).reset_index(drop=True)
        if len(labels) != target.shape[0]:
            raise ValueError(
                f"X tiene {len(labels)} filas y el objetivo {target.shape[0]}."
            )

        work = labels.copy()
        work["__target"] = target

        self.group_quantiles_ = {}
        self.group_sizes_ = {}
        for keys, chunk in work.groupby(list(self.group_columns), dropna=False, observed=True):
            key = keys if isinstance(keys, tuple) else (keys,)
            values = chunk["__target"].to_numpy()
            self.group_sizes_[key] = int(values.size)
            if values.size < self.min_group_size:
                # Grupo demasiado pequeño: sus cuantiles serían ruido, se resuelve con respaldo.
                continue
            self.group_quantiles_[key] = self._empirical_quantiles(values)

        self.fallback_quantiles_ = {}
        self.fallback_sizes_ = {}
        for key, chunk in work.groupby(self.fallback_column, dropna=False, observed=True):
            values = chunk["__target"].to_numpy()
            self.fallback_sizes_[str(key)] = int(values.size)
            if values.size < self.min_group_size:
                continue
            self.fallback_quantiles_[str(key)] = self._empirical_quantiles(values)

        self.global_quantiles_ = self._empirical_quantiles(work["__target"].to_numpy())
        self.global_size_ = int(target.shape[0])
        self.fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("El baseline no ha sido ajustado; llamar fit antes de predict.")

    def _resolve_row(self, key: tuple[str, ...]) -> tuple[dict[str, float], str, int]:
        quantiles = self.group_quantiles_.get(key)
        if quantiles is not None:
            return quantiles, LEVEL_GROUP, self.group_sizes_[key]

        fallback_key = key[self.group_columns.index(self.fallback_column)]
        quantiles = self.fallback_quantiles_.get(fallback_key)
        if quantiles is not None:
            return quantiles, LEVEL_FALLBACK, self.fallback_sizes_[fallback_key]

        # Categorías desconocidas o sin respaldo suficiente: distribución global del entrenamiento.
        return self.global_quantiles_, LEVEL_GLOBAL, self.global_size_

    def predict_with_source(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predice cuantiles y reporta de qué nivel salió cada fila."""

        self._check_fitted()
        labels = self._label_frame(X)
        keys = list(zip(*(labels[column] for column in self.group_columns)))

        rows = []
        for key in keys:
            quantiles, level, size = self._resolve_row(tuple(str(part) for part in key))
            rows.append({**quantiles, "group_level": level, "group_size": size})
        return pd.DataFrame(rows, index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Devuelve p10, p35, p50 y p90 en MXN según el contrato común."""

        return self.predict_with_source(X).loc[:, self.quantile_columns]

    def rules_frame(self) -> pd.DataFrame:
        """Tabla de las reglas ajustadas, para documentar y auditar el baseline."""

        self._check_fitted()
        records: list[dict[str, Any]] = []
        for key, quantiles in self.group_quantiles_.items():
            records.append(
                {
                    "level": LEVEL_GROUP,
                    "key": " | ".join(key),
                    "n": self.group_sizes_[key],
                    **quantiles,
                }
            )
        for key, quantiles in self.fallback_quantiles_.items():
            records.append(
                {"level": LEVEL_FALLBACK, "key": key, "n": self.fallback_sizes_[key], **quantiles}
            )
        records.append(
            {
                "level": LEVEL_GLOBAL,
                "key": LEVEL_GLOBAL,
                "n": self.global_size_,
                **self.global_quantiles_,
            }
        )
        return pd.DataFrame.from_records(records)
