"""Ridge sobre log(precio) con cuantiles globales de residuos internos."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.modeling_data import prepare_features
from src.preprocessing import build_preprocessor

QUANTILE_COLUMNS = ['p10', 'p35', 'p50', 'p90']
QUANTILE_LEVELS = [0.10, 0.35, 0.50, 0.90]


class RidgeQuantileModel(BaseEstimator):
    def __init__(self, alpha=1.0, include_optional=False, min_category_count=20,
                 calibration_splits=3, random_state=42):
        self.alpha = alpha
        self.include_optional = include_optional
        self.min_category_count = min_category_count
        self.calibration_splits = calibration_splits
        self.random_state = random_state

    def _pipeline(self):
        return Pipeline([
            ('preprocess', build_preprocessor(self.include_optional, self.min_category_count)),
            ('ridge', Ridge(alpha=self.alpha, solver='lsqr', tol=1e-6)),
        ])

    def fit(self, X, y, groups=None):
        features = prepare_features(X)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(features) or not np.isfinite(target).all() or (target <= 0).any():
            raise ValueError('y debe contener un precio positivo y finito por fila, en MXN.')
        if self.alpha <= 0 or self.calibration_splits < 2:
            raise ValueError('alpha debe ser positivo y calibration_splits al menos 2.')
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != len(target) or pd.isna(groups).any():
                raise ValueError('groups debe tener un anfitrión válido por fila.')
            n_splits = min(self.calibration_splits, len(np.unique(groups)))
            if n_splits < 2:
                raise ValueError('Se requieren al menos dos anfitriones para calibrar.')
            folds = GroupKFold(n_splits).split(features, groups=groups)
            self.calibration_method_ = 'grouped_oof_log_residuals'
        else:
            n_splits = min(self.calibration_splits, len(target))
            if n_splits < 2:
                raise ValueError('Se requieren al menos dos filas para calibrar.')
            folds = KFold(n_splits, shuffle=True, random_state=self.random_state).split(features)
            self.calibration_method_ = 'row_oof_log_residuals'
        log_target = np.log(target)
        internal_predictions = np.empty(len(target))
        # Cada precio se predice con un pipeline que no vio esa fila ni su objetivo.
        for train, valid in folds:
            model = self._pipeline().fit(features.iloc[train], log_target[train])
            internal_predictions[valid] = model.predict(features.iloc[valid])
        self.residual_quantiles_ = np.quantile(log_target - internal_predictions, QUANTILE_LEVELS)
        self.pipeline_ = self._pipeline().fit(features, log_target)
        self.n_training_rows_ = len(target)
        return self

    def predict(self, X):
        check_is_fitted(self, ['pipeline_', 'residual_quantiles_'])
        center = self.pipeline_.predict(prepare_features(X))
        # Los cuatro desplazamientos ordenados evitan cruces de cuantiles.
        with np.errstate(over='ignore', under='ignore'):
            prices = np.exp(center[:, None] + self.residual_quantiles_)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError('Predicción fuera del rango numérico; revisar las entradas.')
        return pd.DataFrame(prices, columns=QUANTILE_COLUMNS, index=X.index)

    def predict_quantiles(self, X):
        return self.predict(X)
