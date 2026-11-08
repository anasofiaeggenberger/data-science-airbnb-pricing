"""Carga de junio sin aprender imputaciones ni mirar septiembre."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.preparation import DEFAULT_JUNE_PATH, parse_price_mxn, _amenities_count

NUMERIC_FEATURES = ['latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'amenities_count']
CATEGORICAL_FEATURES = ['neighbourhood_cleansed', 'property_type', 'room_type']
OPTIONAL_NUMERIC = ['minimum_nights', 'maximum_nights']
OPTIONAL_CATEGORICAL = ['host_has_profile_pic', 'host_identity_verified']


def feature_columns(include_optional=False):
    numeric = NUMERIC_FEATURES + (OPTIONAL_NUMERIC if include_optional else [])
    categorical = CATEGORICAL_FEATURES + (OPTIONAL_CATEGORICAL if include_optional else [])
    return numeric, categorical


def prepare_features(raw):
    """Conversiones por fila; no calcula estadísticas de la muestra."""
    result = raw.copy()
    if 'amenities_count' not in result and 'amenities' in result:
        result['amenities_count'] = result['amenities'].map(_amenities_count)
    numeric, categorical = feature_columns(True)
    for column in numeric:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    for column in categorical:
        if column not in result:
            result[column] = np.nan
        values = result[column].astype('string').str.strip()
        result[column] = values.replace('', pd.NA).astype(object).where(values.notna() & values.ne(''), np.nan)
    return result[numeric + categorical]


def load_development(path=DEFAULT_JUNE_PATH):
    """Devuelve X, y y metadata alineados. El path alternativo sirve para pruebas."""
    raw = pd.read_csv(Path(path), dtype={'id': 'string', 'host_id': 'string'})
    if not {'id', 'host_id', 'price'}.issubset(raw.columns):
        raise ValueError('Se requieren id, host_id y price del snapshot original.')
    raw['id'] = raw['id'].str.strip()
    raw['host_id'] = raw['host_id'].str.strip()
    raw['price_mxn'] = raw['price'].map(parse_price_mxn)
    valid = raw['id'].notna() & raw['id'].ne('') & raw['price_mxn'].notna()
    raw = raw.loc[valid].sort_values('id').reset_index(drop=True)
    if raw.empty or raw['id'].duplicated().any():
        raise ValueError('Desarrollo vacío o IDs duplicados; revisar la fuente.')
    # No inventar un anfitrión distinto por anuncio: todos los desconocidos se agrupan.
    raw['host_id'] = raw['host_id'].fillna('__unknown_host__').replace('', '__unknown_host__')
    return prepare_features(raw), raw['price_mxn'], raw[['id', 'host_id']].copy()


def make_folds(metadata, n_splits=5):
    """Posiciones para .iloc; todos los modelos deben reutilizar estos folds."""
    groups = metadata['host_id']
    if groups.isna().any() or groups.nunique() < n_splits:
        raise ValueError('Faltan anfitriones válidos para formar las particiones.')
    return list(GroupKFold(n_splits=n_splits).split(metadata, groups=groups))
