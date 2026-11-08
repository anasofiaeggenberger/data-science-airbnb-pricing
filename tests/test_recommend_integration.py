"""Entrenamiento, serialización e inferencia por CLI sin snapshots reales."""
import json
from pathlib import Path
import subprocess
import sys

import joblib
import pandas as pd
import pytest

from src import train_ridge
from src.modeling_data import load_development
from tests.test_ridge_model import sample


@pytest.mark.parametrize('artifact_format', ['bundle', 'model'])
def test_saved_training_model_runs_recommender_cli(tmp_path, monkeypatch, artifact_format):
    repo_root = Path(__file__).resolve().parents[1]
    X, y, groups = sample()
    source = tmp_path / 'synthetic_development.csv'
    X.assign(id=range(len(X)), host_id=groups, price=y).to_csv(source, index=False)
    (tmp_path / 'docs').mkdir()

    # Ejecutar el guardado real del entrenamiento, limitado a datos sintéticos.
    monkeypatch.setattr(train_ridge, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(train_ridge, 'DEFAULT_JUNE_PATH', source)
    monkeypatch.setattr(train_ridge, 'load_development', lambda: load_development(source))
    monkeypatch.setattr(sys, 'argv', ['train_ridge'])
    train_ridge.main()

    model_path = tmp_path / 'models/ridge.joblib'
    saved = joblib.load(model_path)
    report = json.loads((tmp_path / 'docs/ridge_validation.json').read_text())
    assert saved['metadata'] == report
    assert report['test_evaluated'] is False
    assert report['n_rows'] == len(X)
    assert report['metrics_status'] == 'provisional_pending_shared_metrics'
    assert report['metrics_implementation'] == 'sklearn_metrics_and_local_interval_calculations'

    if artifact_format == 'model':
        # El recomendador admite también un modelo sin envoltorio de metadatos.
        model_path = tmp_path / 'model_only.joblib'
        joblib.dump(saved['model'], model_path)

    listing_path = repo_root / 'examples/listing.json'
    listing = json.loads(listing_path.read_text())
    expected = saved['model'].predict(pd.DataFrame([listing])).iloc[0]
    completed = subprocess.run(
        [sys.executable, '-m', 'src.recommend', '--model', str(model_path),
         '--listing', str(listing_path)],
        cwd=repo_root, capture_output=True, text=True, check=True, timeout=30,
    )
    result = json.loads(completed.stdout)
    assert result['currency'] == 'MXN'
    assert result['nominal_coverage'] == 0.8
    for output, column in [('lower_price', 'p10'), ('recommended_price', 'p35'),
                           ('median_price', 'p50'), ('upper_price', 'p90')]:
        assert result[output] == round(float(expected[column]), 2)
    assert 0 < result['lower_price'] <= result['recommended_price'] <= result['median_price'] <= result['upper_price']
    if artifact_format == 'bundle':
        assert result['model_status'] == report['status']
    else:
        assert 'model_status' not in result
