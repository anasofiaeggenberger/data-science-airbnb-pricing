"""Pipeline basado en los ejemplos de Feature Engineering y Proyecto 1."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.modeling_data import feature_columns


def build_preprocessor(include_optional=False, min_category_count=20):
    numeric, categorical = feature_columns(include_optional)
    numbers = Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    categories = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Missing', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                 min_frequency=min_category_count, sparse_output=True)),
    ])
    return ColumnTransformer([('numeric', numbers, numeric),
                              ('categorical', categories, categorical)], remainder='drop')
