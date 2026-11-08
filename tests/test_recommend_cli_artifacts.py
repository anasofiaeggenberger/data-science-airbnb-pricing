"""El recomendador por CLI sobre artefactos serializados de Ridge y LightGBM.

Todo se construye con datos sintéticos: estas pruebas nunca abren los snapshots
reales de junio ni de septiembre.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluate_final import file_sha256, train_and_serialize_winner


REPO_ROOT = Path(__file__).resolve().parents[1]

LISTING = {
    "latitude": 19.41,
    "longitude": -99.16,
    "accommodates": 3,
    "bathrooms": 1,
    "bedrooms": 1,
    "beds": 2,
    "neighbourhood_cleansed": "Centro",
    "property_type": "Apartment",
    "room_type": "Entire home/apt",
}

CONFIGURATIONS = {
    "ridge": {"alpha": 1.0},
    "lightgbm": {
        "n_estimators": 40,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_child_samples": 10,
    },
}


def write_synthetic_development(path: Path, n: int = 240, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    pd.DataFrame(
        {
            "id": [f"L{index}" for index in range(n)],
            "host_id": [f"H{index % 30}" for index in range(n)],
            "price": [f"${700 + index * 7}.00" for index in range(n)],
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
    ).to_csv(path, index=False, compression="gzip")
    return path


def frozen_for(model_name: str, june: Path) -> dict:
    return {
        "model": model_name,
        "params": CONFIGURATIONS[model_name],
        "include_optional": False,
        "price_cap_quantile": None,
        "quantiles": {"p10": 0.10, "p35": 0.35, "p50": 0.50, "p90": 0.90},
        "selection_metric": "pinball_recommended",
        "june_sha256": file_sha256(june),
        "fold_assignment_sha256": "sintetico",
        "development_metrics": {},
    }


@pytest.mark.parametrize("model_name", ["ridge", "lightgbm"])
def test_recommend_cli_runs_on_a_serialized_artifact(tmp_path: Path, model_name: str) -> None:
    june = write_synthetic_development(tmp_path / "development.csv.gz")
    artifact = tmp_path / f"final_{model_name}.joblib"
    train_and_serialize_winner(frozen_for(model_name, june), june, artifact)

    listing_path = tmp_path / "listing.json"
    listing_path.write_text(json.dumps(LISTING), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "src.recommend", "--model", str(artifact),
         "--listing", str(listing_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=120,
    )
    result = json.loads(completed.stdout)

    assert result["currency"] == "MXN"
    assert result["model_status"] == "frozen_development_winner"

    # Los cuatro cuantiles del contrato, en el orden que exige el rango.
    quantiles = [
        result["lower_price"],      # p10
        result["recommended_price"],  # p35
        result["median_price"],     # p50
        result["upper_price"],      # p90
    ]
    for value in quantiles:
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert value > 0
    assert quantiles == sorted(quantiles)


@pytest.mark.parametrize("model_name", ["ridge", "lightgbm"])
def test_cli_output_matches_the_artifact_predictions(tmp_path: Path, model_name: str) -> None:
    june = write_synthetic_development(tmp_path / "development.csv.gz")
    artifact = tmp_path / f"final_{model_name}.joblib"
    train_and_serialize_winner(frozen_for(model_name, june), june, artifact)

    listing_path = tmp_path / "listing.json"
    listing_path.write_text(json.dumps(LISTING), encoding="utf-8")

    expected = joblib.load(artifact)["model"].predict(pd.DataFrame([LISTING])).iloc[0]
    completed = subprocess.run(
        [sys.executable, "-m", "src.recommend", "--model", str(artifact),
         "--listing", str(listing_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=120,
    )
    result = json.loads(completed.stdout)

    # El proceso aparte reproduce exactamente lo que predice el artefacto cargado.
    for output_key, quantile in [
        ("lower_price", "p10"),
        ("recommended_price", "p35"),
        ("median_price", "p50"),
        ("upper_price", "p90"),
    ]:
        assert result[output_key] == round(float(expected[quantile]), 2)
