"""Pruebas del evaluador final con datos sintéticos, antes de tocar septiembre."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate_final import (
    file_sha256,
    freeze_solution,
    load_test_set,
    main,
    run_final_evaluation,
    train_and_serialize_winner,
    validate_frozen_configuration,
)


def write_listings(path: Path, ids: list[str], prices: list[str], seed: int = 0) -> Path:
    """Snapshot sintético con las columnas que el pipeline necesita."""

    n = len(ids)
    rng = np.random.default_rng(seed)
    pd.DataFrame(
        {
            "id": ids,
            "host_id": [f"H{index % 25}" for index in range(n)],
            "price": prices,
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


@pytest.fixture
def snapshots(tmp_path: Path) -> tuple[Path, Path]:
    june_ids = [f"L{index}" for index in range(200)]
    june_prices = [f"${800 + index * 3}.00" for index in range(200)]
    june = write_listings(tmp_path / "june.csv.gz", june_ids, june_prices)

    # Septiembre repite L100-L199 y agrega L200-L249; tres de los nuevos no tienen precio válido.
    september_ids = [f"L{index}" for index in range(100, 250)]
    september_prices = [f"${900 + index * 2}.00" for index in range(100, 250)]
    september_prices[-1] = ""
    september_prices[-2] = "$0.00"
    september_prices[-3] = "sin precio"
    september = write_listings(tmp_path / "september.csv.gz", september_ids, september_prices, 1)
    return june, september


def frozen_record(june: Path) -> dict:
    return {
        "model": "baseline",
        "params": {"min_group_size": 20},
        "include_optional": False,
        "price_cap_quantile": None,
        "quantiles": {"p10": 0.10, "p35": 0.35, "p50": 0.50, "p90": 0.90},
        "selection_metric": "pinball_recommended",
        "june_sha256": file_sha256(june),
        "fold_assignment_sha256": "n/a",
        "development_metrics": {},
    }


def test_test_set_keeps_only_new_ids_with_a_valid_price(snapshots) -> None:
    june, september = snapshots
    X, y, metadata = load_test_set(june, september)

    assert len(X) == 47  # 50 IDs nuevos menos los tres sin precio válido
    assert metadata["id"].tolist() == sorted(metadata["id"].tolist())
    assert set(metadata["id"]) <= {f"L{index}" for index in range(200, 250)}
    assert (y > 0).all()
    assert metadata["id"].is_unique


def test_test_set_fails_loudly_when_nothing_is_eligible(tmp_path: Path) -> None:
    ids = [f"L{index}" for index in range(30)]
    prices = [f"${500 + index}.00" for index in range(30)]
    june = write_listings(tmp_path / "a.csv.gz", ids, prices)
    september = write_listings(tmp_path / "b.csv.gz", ids, prices)
    with pytest.raises(ValueError, match="vacío"):
        load_test_set(june, september)


def test_freeze_writes_the_winner_without_touching_the_test(tmp_path: Path) -> None:
    comparison = {
        "winner": "lightgbm",
        "include_optional": False,
        "quantiles": {"p10": 0.10, "p35": 0.35, "p50": 0.50, "p90": 0.90},
        "selection_metric": "pinball_recommended",
        "fold_assignment_sha256": "abc",
        "june_sha256": "def",
        "models": [
            {"model": "baseline", "selected_params": {"min_group_size": 20}, "mean": {"a": 1.0}},
            {"model": "lightgbm", "selected_params": {"num_leaves": 31}, "mean": {"a": 0.5}},
        ],
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison), encoding="utf-8")
    output = tmp_path / "frozen.json"

    frozen = freeze_solution(path, output)
    assert frozen["model"] == "lightgbm"
    assert frozen["params"] == {"num_leaves": 31}
    assert frozen["test_evaluated"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["model"] == "lightgbm"

    # Se puede congelar otro modelo a propósito, pero no uno que no se comparó.
    assert freeze_solution(path, None, model="baseline")["model"] == "baseline"
    with pytest.raises(ValueError):
        freeze_solution(path, None, model="red neuronal")


def test_final_evaluation_runs_end_to_end_on_synthetic_snapshots(snapshots) -> None:
    june, september = snapshots
    outcome = run_final_evaluation(frozen_record(june), june, september)
    record = outcome["record"]

    assert record["test_evaluated"] is True
    assert record["n_test_rows"] == 47
    assert record["n_development_rows"] == 200
    assert record["overall"]["n_observations"] == 47
    assert record["overall"]["quantile_crossing_rate"] == pytest.approx(0.0)
    # La evidencia por segmentos viene con tamaños de muestra.
    segments = pd.DataFrame(record["segments"])
    assert set(segments["segment"]) == {"neighbourhood_cleansed", "room_type", "price_band"}
    assert "n" in segments.columns


def test_a_different_june_snapshot_stops_the_evaluation(snapshots, tmp_path: Path) -> None:
    june, september = snapshots
    frozen = frozen_record(june)
    frozen["june_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="no coincide"):
        run_final_evaluation(frozen, june, september)


def test_price_cap_from_development_does_not_break_the_evaluation(snapshots) -> None:
    june, september = snapshots
    frozen = frozen_record(june)
    frozen["price_cap_quantile"] = 0.99
    record = run_final_evaluation(frozen, june, september)["record"]
    # El tope afecta el entrenamiento; el test se evalúa con sus precios originales.
    assert record["n_test_rows"] == 47


def test_cli_refuses_to_touch_september_without_explicit_confirmation(capsys) -> None:
    assert main(["run"]) == 1
    assert "--confirm-frozen" in capsys.readouterr().out


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

WINNER_CONFIGURATIONS = {
    "ridge": {"alpha": 1.0},
    "lightgbm": {
        "n_estimators": 40,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_child_samples": 10,
    },
}


@pytest.mark.parametrize("model_name", ["ridge", "lightgbm"])
def test_winner_is_retrained_serialized_and_usable_by_the_recommender(
    snapshots, tmp_path: Path, model_name: str
) -> None:
    import joblib

    from src.recommend import recommend

    june, _ = snapshots
    frozen = frozen_record(june)
    frozen["model"] = model_name
    frozen["params"] = WINNER_CONFIGURATIONS[model_name]

    artifact_path = tmp_path / f"final_{model_name}.joblib"
    result = train_and_serialize_winner(frozen, june, artifact_path)

    assert artifact_path.exists()
    saved = joblib.load(artifact_path)
    assert set(saved) == {"model", "metadata"}
    metadata = saved["metadata"]
    assert metadata["test_evaluated"] is False
    assert metadata["model"] == model_name
    assert metadata["n_development_rows"] == 200
    # La política de variables viaja con el artefacto.
    assert "neighbourhood_cleansed" in metadata["feature_policy"]["categorical"]
    assert "accommodates" in metadata["feature_policy"]["numeric"]
    assert metadata["development_snapshot"] == "2026-06-15"

    # El artefacto sirve al recomendador de Víctor sin adaptaciones.
    quote = recommend(saved["model"], LISTING)
    assert quote["currency"] == "MXN"
    assert 0 < quote["lower_price"] <= quote["recommended_price"] <= quote["median_price"]
    assert quote["median_price"] <= quote["upper_price"]
    assert result["path"] == artifact_path


def test_serialization_refuses_a_configuration_the_code_cannot_build(snapshots, tmp_path) -> None:
    june, _ = snapshots
    frozen = frozen_record(june)
    frozen["model"] = "red neuronal"
    with pytest.raises(ValueError, match="construible"):
        train_and_serialize_winner(frozen, june, tmp_path / "x.joblib")


def test_frozen_configuration_is_validated_before_anything_runs(snapshots) -> None:
    june, _ = snapshots

    incomplete = {"model": "baseline", "params": {}}
    with pytest.raises(ValueError, match="no tiene"):
        validate_frozen_configuration(incomplete)

    already_evaluated = frozen_record(june) | {"test_evaluated": True}
    with pytest.raises(ValueError, match="ya fue evaluada"):
        validate_frozen_configuration(already_evaluated)

    # Un contrato de cuantiles distinto haría incomparables los resultados.
    other_contract = frozen_record(june)
    other_contract["quantiles"] = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p90": 0.90}
    with pytest.raises(ValueError, match="contrato de cuantiles"):
        validate_frozen_configuration(other_contract)


def test_cli_refuses_to_repeat_an_evaluation_that_already_exists(tmp_path, capsys) -> None:
    results = tmp_path / "final_evaluation.json"
    results.write_text("{}", encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    frozen.write_text("{}", encoding="utf-8")
    september = tmp_path / "september.csv.gz"
    write_listings(september, ["L1"], ["$900.00"])

    code = main(
        [
            "run",
            "--confirm-frozen",
            "--frozen", str(frozen),
            "--september", str(september),
            "--results", str(results),
        ]
    )
    assert code == 1
    assert "ya se ejecutó" in capsys.readouterr().out
