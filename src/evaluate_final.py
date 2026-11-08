"""Evaluación final sobre el test temporal. Se ejecuta una sola vez, tras el congelamiento."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.audit_data import ids_from_file
from src.metrics import QUANTILE_LEVELS, SELECTION_METRIC, evaluate_quantile_predictions
from src.modeling_data import feature_columns, load_development, prepare_features
from src.preparation import (
    DEFAULT_JUNE_PATH,
    DEFAULT_SEPTEMBER_PATH,
    REPO_ROOT,
    parse_price_mxn,
)
from src.segments import build_segment_tables
from src.train_models import QUANTILE_COLUMNS, build_model

DEFAULT_COMPARISON_PATH = REPO_ROOT / "docs" / "model_comparison.json"
DEFAULT_FROZEN_PATH = REPO_ROOT / "docs" / "frozen_solution.json"
DEFAULT_RESULTS_PATH = REPO_ROOT / "docs" / "final_evaluation.json"
DEFAULT_TEST_PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "test_predictions.csv"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"

UNKNOWN_HOST = "__unknown_host__"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_test_set(
    june_path: Path = DEFAULT_JUNE_PATH,
    september_path: Path = DEFAULT_SEPTEMBER_PATH,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Reconstruye el test con la regla original: IDs nuevos en septiembre y precio válido."""

    june_ids = ids_from_file(Path(june_path))
    raw = pd.read_csv(
        Path(september_path), dtype={"id": "string", "host_id": "string"}, low_memory=False
    )
    if not {"id", "host_id", "price"}.issubset(raw.columns):
        raise ValueError("Se requieren id, host_id y price del snapshot original.")

    raw["id"] = raw["id"].str.strip()
    raw["host_id"] = raw["host_id"].str.strip()
    raw["price_mxn"] = raw["price"].map(parse_price_mxn)
    # Mismas dos condiciones del manifiesto: ausente en junio y precio válido.
    is_new = ~raw["id"].isin(june_ids)
    is_valid = raw["id"].notna() & raw["id"].ne("") & raw["price_mxn"].notna()
    test = raw.loc[is_new & is_valid].sort_values("id").reset_index(drop=True)
    if test.empty:
        raise ValueError("El test reconstruido quedó vacío; revisar los snapshots.")
    test["host_id"] = test["host_id"].fillna(UNKNOWN_HOST).replace("", UNKNOWN_HOST)
    return prepare_features(test), test["price_mxn"], test[["id", "host_id"]].copy()


def freeze_solution(
    comparison_path: Path = DEFAULT_COMPARISON_PATH,
    output_path: Path | None = DEFAULT_FROZEN_PATH,
    model: str | None = None,
) -> dict[str, Any]:
    """Guarda la configuración ganadora antes de mirar septiembre."""

    comparison = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    chosen = model or comparison["winner"]
    records = {record["model"]: record for record in comparison["models"]}
    if chosen not in records:
        raise ValueError(f"El modelo '{chosen}' no está en la comparación.")

    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": chosen,
        "params": records[chosen]["selected_params"],
        "include_optional": comparison["include_optional"],
        "price_cap_quantile": comparison.get("price_cap_quantile"),
        "quantiles": comparison["quantiles"],
        "selection_metric": comparison["selection_metric"],
        "development_metrics": records[chosen]["mean"],
        "fold_assignment_sha256": comparison["fold_assignment_sha256"],
        "june_sha256": comparison["june_sha256"],
        "source_comparison": str(Path(comparison_path).name),
        "test_evaluated": False,
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return frozen


def validate_frozen_configuration(frozen: dict[str, Any]) -> None:
    """Comprueba que la configuración congelada sea usable por el código actual."""

    missing = {"model", "params", "include_optional", "quantiles"} - set(frozen)
    if missing:
        raise ValueError(f"La solución congelada no tiene {sorted(missing)}.")
    if frozen.get("test_evaluated"):
        raise ValueError("Esa solución ya fue evaluada contra el test; no se repite.")
    if dict(frozen["quantiles"]) != dict(QUANTILE_LEVELS):
        # Evaluar con otro contrato de cuantiles haría incomparables los resultados.
        raise ValueError("El contrato de cuantiles congelado no coincide con src/metrics.py.")
    try:
        build_model(frozen["model"], dict(frozen["params"]), frozen["include_optional"])
    except Exception as error:
        raise ValueError(f"La configuración congelada ya no es construible: {error}") from error


def _fit_frozen_model(frozen: dict[str, Any], X: pd.DataFrame, y: pd.Series, groups: Any) -> Any:
    """Entrena la configuración congelada, aplicando su tope de precios si lo define."""

    y_fit = y
    cap_quantile = frozen.get("price_cap_quantile")
    if cap_quantile is not None:
        # El tope se aprende con desarrollo, nunca con el test.
        y_fit = y.clip(upper=float(y.quantile(cap_quantile)))
    model = build_model(frozen["model"], dict(frozen["params"]), frozen["include_optional"])
    return model.fit(X, y_fit, groups=groups)


def _check_quantile_contract(predictions: pd.DataFrame) -> None:
    values = predictions.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float)
    if not pd.notna(values).all() or (values <= 0).any():
        raise ValueError("El artefacto produjo cuantiles nulos o no positivos.")
    if (values[:, 1:] < values[:, :-1]).any():
        raise ValueError("El artefacto produjo cuantiles cruzados.")


def train_and_serialize_winner(
    frozen: dict[str, Any],
    june_path: Path = DEFAULT_JUNE_PATH,
    output_path: Path | None = None,
    sample_rows: int = 200,
) -> dict[str, Any]:
    """Reentrena la solución congelada con todo el desarrollo y guarda el artefacto joblib.

    El artefacto queda en el formato que consume `src/recommend.py`: un diccionario
    con `model` y `metadata`. No abre el snapshot de septiembre.
    """

    validate_frozen_configuration(frozen)
    X, y, metadata = load_development(june_path)
    model = _fit_frozen_model(frozen, X, y, metadata["host_id"])

    # Contrato verificado sobre desarrollo antes de guardar: positivo y ordenado.
    _check_quantile_contract(model.predict(X.head(min(sample_rows, len(X)))))

    numeric, categorical = feature_columns(frozen["include_optional"])
    artifact_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_development_winner",
        "test_evaluated": False,
        "development_snapshot": "2026-06-15",
        "n_development_rows": int(len(X)),
        "model": frozen["model"],
        "params": frozen["params"],
        "include_optional": frozen["include_optional"],
        "price_cap_quantile": frozen.get("price_cap_quantile"),
        "feature_policy": {"numeric": list(numeric), "categorical": list(categorical)},
        "quantiles": frozen["quantiles"],
        "selection_metric": frozen.get("selection_metric"),
        "development_metrics": frozen.get("development_metrics"),
        "fold_assignment_sha256": frozen.get("fold_assignment_sha256"),
        "june_sha256": frozen.get("june_sha256"),
        "python_version": sys.version.split()[0],
        "limitation": (
            "Entrenado solo con junio. Estima precios publicados comparables; no promete "
            "reservas, ingresos ni cobertura individual del intervalo."
        ),
    }

    if output_path is None:
        output_path = DEFAULT_MODEL_DIR / f"final_{frozen['model']}.joblib"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": artifact_metadata}, Path(output_path))
    return {"model": model, "metadata": artifact_metadata, "path": Path(output_path)}


def run_final_evaluation(
    frozen: dict[str, Any],
    june_path: Path = DEFAULT_JUNE_PATH,
    september_path: Path = DEFAULT_SEPTEMBER_PATH,
) -> dict[str, Any]:
    """Entrena la solución congelada con todo junio y la evalúa una vez sobre el test."""

    X_dev, y_dev, metadata_dev = load_development(june_path)
    if frozen.get("june_sha256") and Path(june_path).exists():
        # El desarrollo debe ser byte a byte el mismo con el que se congeló.
        if file_sha256(june_path) != frozen["june_sha256"]:
            raise ValueError(
                "El snapshot de junio no coincide con el que se usó para congelar la solución."
            )

    validate_frozen_configuration(frozen)
    model = _fit_frozen_model(frozen, X_dev, y_dev, metadata_dev["host_id"])

    X_test, y_test, metadata_test = load_test_set(june_path, september_path)
    predictions = model.predict(X_test)
    overall = evaluate_quantile_predictions(y_test, predictions)

    frame = metadata_test.assign(
        price_mxn=y_test.to_numpy(),
        neighbourhood_cleansed=X_test["neighbourhood_cleansed"].to_numpy(),
        room_type=X_test["room_type"].to_numpy(),
    ).join(predictions)
    segments = build_segment_tables(frame)

    record = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": True,
        "development_snapshot": "2026-06-15",
        "test_snapshot": "2026-09-23",
        "test_rule": "IDs presentes en septiembre, ausentes en junio y con precio válido.",
        "n_development_rows": int(len(X_dev)),
        "n_test_rows": int(len(X_test)),
        "frozen_solution": {key: value for key, value in frozen.items() if key != "test_evaluated"},
        "overall": overall,
        "segments": segments.to_dict(orient="records"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "limitation": (
            "Una sola evaluación sobre el test temporal. No se reajusta nada después de verla. "
            "Los anuncios nuevos del snapshot no son necesariamente anuncios recién publicados."
        ),
    }
    return {"record": record, "predictions": frame, "segments": segments}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="Guardar la solución ganadora antes del test.")
    freeze.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON_PATH)
    freeze.add_argument("--output", type=Path, default=DEFAULT_FROZEN_PATH)
    freeze.add_argument("--model", type=str, default=None)
    freeze.add_argument("--june", type=Path, default=DEFAULT_JUNE_PATH)
    freeze.add_argument("--artifact", type=Path, default=None)
    freeze.add_argument(
        "--skip-artifact",
        action="store_true",
        help="Solo guardar la configuración, sin reentrenar ni serializar el modelo.",
    )

    run = sub.add_parser("run", help="Evaluar una sola vez sobre septiembre.")
    run.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN_PATH)
    run.add_argument("--june", type=Path, default=DEFAULT_JUNE_PATH)
    run.add_argument("--september", type=Path, default=DEFAULT_SEPTEMBER_PATH)
    run.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    run.add_argument("--predictions", type=Path, default=DEFAULT_TEST_PREDICTIONS_PATH)
    run.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="Confirma que el equipo congeló la solución. Sin esto no se toca septiembre.",
    )
    run.add_argument(
        "--allow-rerun",
        action="store_true",
        help="Permite sobrescribir una evaluación final ya ejecutada. Normalmente no se usa.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "freeze":
        frozen = freeze_solution(args.comparison, args.output, args.model)
        print(f"Solución congelada: {frozen['model']} {frozen['params']}")
        print(f"Configuración en {args.output}.")
        if args.skip_artifact:
            print("Artefacto no generado (--skip-artifact).")
        else:
            artifact = train_and_serialize_winner(frozen, args.june, args.artifact)
            print(f"Artefacto entrenado con {artifact['metadata']['n_development_rows']} filas "
                  f"de junio y guardado en {artifact['path']}.")
            print("Uso: uv run python -m src.recommend --model "
                  f"{artifact['path']} --listing examples/listing.json")
        print("Septiembre sigue sin evaluarse.")
        return 0

    if not args.confirm_frozen:
        # Candado del protocolo: la evaluación real exige una confirmación explícita.
        print(
            "La evaluación final necesita --confirm-frozen.\n"
            "Antes de usarla: la solución debe estar congelada, documentada y revisada.\n"
            "Esta evaluación se ejecuta una sola vez y después no se reajusta nada."
        )
        return 1
    if not Path(args.frozen).exists():
        print(f"No existe {args.frozen}. Ejecutar primero: evaluate_final.py freeze")
        return 1
    if not Path(args.september).exists():
        print(f"No existe el snapshot de test en {args.september}.")
        return 1
    if Path(args.results).exists() and not args.allow_rerun:
        # El test es de un solo uso: repetirlo invalidaría su carácter de estimación independiente.
        print(
            f"Ya existe {args.results}: esta evaluación final ya se ejecutó.\n"
            "El test temporal se evalúa una sola vez. Si de verdad hay que repetirla, "
            "documentar el motivo y usar --allow-rerun."
        )
        return 1

    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    outcome = run_final_evaluation(frozen, args.june, args.september)

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).write_text(
        json.dumps(outcome["record"], ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    outcome["predictions"].to_csv(args.predictions, index=False)

    overall = outcome["record"]["overall"]
    print(f"Filas de test: {outcome['record']['n_test_rows']}")
    for key in (SELECTION_METRIC, "mae_recommended", "coverage", "mean_interval_width"):
        print(f"  {key:22s} {overall[key]:12.4f}")
    print(f"\nResultados en {args.results}. No reajustar nada después de esto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
