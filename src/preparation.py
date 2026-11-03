"""Temporal split, feature policy, and development-only cleaning utilities."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.airbnb_data import parse_price
from src.audit_data import ids_from_file


REPO_ROOT = Path(__file__).resolve().parents[1]
JUNE_SNAPSHOT_DATE = "2026-06-15"
SEPTEMBER_SNAPSHOT_DATE = "2026-09-23"
DEFAULT_JUNE_PATH = REPO_ROOT / "data" / "raw" / "mexico_city_2026-06-15_listings.csv.gz"
DEFAULT_SEPTEMBER_PATH = REPO_ROOT / "data" / "raw" / "mexico_city_2026-09-23_listings.csv.gz"
DEFAULT_PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "development_clean.csv"
DEFAULT_SPLIT_MANIFEST = REPO_ROOT / "docs" / "split_manifest.json"
DEFAULT_CLEANING_LOG = REPO_ROOT / "docs" / "cleaning_log.json"
DEFAULT_FEATURE_POLICY = REPO_ROOT / "docs" / "feature_policy.md"

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
]
CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "property_type",
    "room_type",
    "host_has_profile_pic",
    "host_identity_verified",
]
DEVELOPMENT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["amenities_count"]


@dataclass
class PreparedDevelopment:
    """In-memory result of the June-only cleaning pipeline."""

    data: pd.DataFrame
    cleaning_log: dict[str, Any]
    feature_policy: list[dict[str, str]]


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _is_blank(value: Any) -> bool:
    if value is None or value.__class__.__name__ in {"NAType", "NaTType"}:
        return True
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def price_status(value: Any) -> str:
    """Classify a raw price without using any test information."""

    if _is_blank(value):
        return "missing"
    parsed = parse_price(value)
    if parsed is None or not math.isfinite(parsed):
        return "invalid"
    if parsed <= 0:
        return "zero_or_negative"
    return "valid"


def parse_price_mxn(value: Any) -> float:
    """Parse the documented local-currency price as MXN for Mexico City."""

    parsed = parse_price(value)
    if parsed is None or not math.isfinite(parsed) or parsed <= 0:
        return float("nan")
    return float(parsed)


def _iter_id_price_chunks(path: Path, chunksize: int = 50_000):
    columns = list(pd.read_csv(path, compression="infer", nrows=0).columns)
    missing = {column for column in ("id", "price") if column not in columns}
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en {path}: {sorted(missing)}")
    return pd.read_csv(
        path,
        compression="infer",
        usecols=["id", "price"],
        dtype={"id": "string", "price": "string"},
        chunksize=chunksize,
    )


def _price_status_counts(path: Path, chunksize: int = 50_000) -> dict[str, int]:
    counts = {"missing": 0, "invalid": 0, "zero_or_negative": 0, "valid": 0}
    rows = 0
    for chunk in _iter_id_price_chunks(path, chunksize=chunksize):
        rows += len(chunk)
        for value in chunk["price"].tolist():
            counts[price_status(value)] += 1
    counts["rows"] = rows
    return counts


def build_temporal_split_manifest(
    development_path: Path = DEFAULT_JUNE_PATH,
    recent_path: Path = DEFAULT_SEPTEMBER_PATH,
    output_path: Path | None = DEFAULT_SPLIT_MANIFEST,
) -> dict[str, Any]:
    """Count temporal eligibility without writing test rows or test IDs."""

    june_ids = ids_from_file(development_path)
    development_prices = _price_status_counts(development_path)
    new_ids: set[str] = set()
    new_statuses: dict[str, str] = {}
    recent_id_null_count = 0
    recent_row_count = 0

    for chunk in _iter_id_price_chunks(recent_path):
        recent_row_count += len(chunk)
        for listing_id, raw_price in zip(chunk["id"].tolist(), chunk["price"].tolist()):
            if _is_blank(listing_id):
                recent_id_null_count += 1
                continue
            listing_id = str(listing_id)
            if listing_id not in june_ids and listing_id not in new_ids:
                new_ids.add(listing_id)
                new_statuses[listing_id] = price_status(raw_price)

    eligible_new = sum(status == "valid" for status in new_statuses.values())
    excluded_new = len(new_statuses) - eligible_new
    exclusion_by_status = {
        status: sum(value == status for value in new_statuses.values())
        for status in ("missing", "invalid", "zero_or_negative")
    }
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development": {
            "snapshot_date": JUNE_SNAPSHOT_DATE,
            "path": _relative(development_path),
            "definition": "Todas las filas del snapshot de junio; la limpieza de modelado conserva solo precio válido.",
            "row_count": development_prices["rows"],
            "unique_id_count": len(june_ids),
            "id_null_count": development_prices["rows"] - len(june_ids),
            "price_status_counts": development_prices,
        },
        "test": {
            "snapshot_date": SEPTEMBER_SNAPSHOT_DATE,
            "path": _relative(recent_path),
            "definition": "IDs presentes en septiembre y ausentes en junio.",
            "recent_row_count": recent_row_count,
            "recent_id_null_count": recent_id_null_count,
            "new_id_count": len(new_ids),
            "eligible_new_id_count": eligible_new,
            "excluded_new_id_count": excluded_new,
            "excluded_new_by_price_status": exclusion_by_status,
            "price_validity_rule": "parseable, finito y estrictamente mayor que cero",
            "records_saved": False,
            "ids_saved": False,
        },
        "isolation": {
            "notebook_loads_test": False,
            "test_used_for_feature_selection": False,
            "test_used_for_cleaning_rules": False,
            "test_used_for_hyperparameters": False,
            "test_use_policy": "reservar para una única evaluación final",
            "interpretation_note": "Nuevo en el snapshot significa nuevo para la observación; no necesariamente recién publicado en Airbnb.",
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _policy_entry(column: str, decision: str, reason: str) -> dict[str, str]:
    return {"column": column, "decision": decision, "reason": reason}


def feature_policy(columns: list[str]) -> list[dict[str, str]]:
    """Return an explicit decision for every column in the detailed file."""

    allowed = {
        "neighbourhood_cleansed": "Alcaldía geocodificada a partir de la ubicación; conocida antes de publicar.",
        "neighbourhood_group_cleansed": "Contexto geográfico derivado de ubicación; se conserva si tiene cobertura.",
        "property_type": "Tipo de propiedad declarado en el anuncio.",
        "room_type": "Tipo de alojamiento declarado en el anuncio.",
        "latitude": "Ubicación aproximada; conocida antes de publicar, con la limitación de anonimización espacial.",
        "longitude": "Ubicación aproximada; conocida antes de publicar, con la limitación de anonimización espacial.",
        "accommodates": "Capacidad máxima declarada.",
        "bathrooms": "Cantidad de baños declarada; se prioriza sobre bathrooms_text para modelar.",
        "bedrooms": "Cantidad de dormitorios declarada.",
        "beds": "Cantidad de camas declarada.",
        "amenities": "Amenidades declaradas antes de publicar; se usará una representación simple en esta fase.",
        "minimum_nights": "Regla de estadía que el anfitrión puede definir antes de publicar; se conserva como candidata opcional porque depende de la estrategia que el anfitrión decida ofrecer.",
        "maximum_nights": "Regla de estadía que el anfitrión puede definir antes de publicar; se conserva como candidata opcional porque depende de la estrategia que el anfitrión decida ofrecer.",
        "instant_bookable": "Configuración del anuncio disponible antes de una reserva.",
        "host_has_profile_pic": "Puede conocerse antes de publicar si el anfitrión completa su perfil, pero el snapshot observa el estado actual; se conserva como candidata opcional y debe probarse sin asumir estabilidad histórica.",
        "host_identity_verified": "Puede conocerse antes de publicar si el anfitrión completa la verificación, pero el snapshot observa el estado actual; se conserva como candidata opcional y debe probarse sin asumir estabilidad histórica.",
    }
    deferred = {
        "neighbourhood": "Texto de ubicación declarado por el anfitrión; puede ser inconsistente, se prefiere neighbourhood_cleansed.",
        "neighbourhood_group_cleansed": "Permitida si hay cobertura; se revisará su redundancia con la alcaldía.",
        "bathrooms_text": "Información potencialmente útil, pero redundante con bathrooms y textual; no entra en la tabla limpia.",
        "host_since": "Conocida por la plataforma, pero funciona como proxy de experiencia y no es estable para un anfitrión nuevo.",
        "host_verifications": "Estado previo y disponible, pero requiere codificación de lista; se difiere para evitar decisiones prematuras.",
        "license": "Dato regulatorio potencialmente previo, pero con alta ausencia y significado heterogéneo; se difiere.",
        "name": "Texto escrito antes de publicar; se difiere NLP para no introducir decisiones no justificadas.",
        "description": "Texto escrito antes de publicar; se difiere NLP para no introducir decisiones no justificadas.",
        "neighborhood_overview": "Texto escrito antes de publicar; se difiere NLP para no introducir decisiones no justificadas.",
        "host_about": "Texto del perfil; se difiere por posible información personal y por no ser necesario para el baseline.",
    }
    prohibited_exact = {
        "id": "Identificador técnico; nunca debe ser predictor.",
        "listing_url": "Identificador/enlace del anuncio; no es una característica causal y puede filtrar identidad.",
        "host_id": "Identificador técnico del anfitrión; puede memorizar el objetivo y no generaliza a un anfitrión nuevo.",
        "host_url": "Enlace identificador del anfitrión; no es predictor.",
        "host_profile_id": "Identificador técnico del perfil; no es predictor.",
        "host_profile_url": "Enlace identificador del perfil; no es predictor.",
        "host_name": "Nombre del anfitrión; no es una característica de publicación apropiada y puede contener información personal.",
        "host_location": "Ubicación autodeclarada del anfitrión, no del alojamiento; no es necesaria para la decisión.",
        "host_neighbourhood": "Zona del anfitrión, no del anuncio; no representa la ubicación del alojamiento.",
        "host_listings_count": "Inventario actual del anfitrión; no representa a un anfitrión nuevo y puede reflejar actividad posterior.",
        "host_total_listings_count": "Inventario actual del anfitrión; no representa a un anfitrión nuevo y puede reflejar actividad posterior.",
        "host_response_time": "Métrica de respuesta posterior a interacciones; fuga de información para publicar.",
        "host_response_rate": "Métrica de comportamiento posterior; fuga de información para publicar.",
        "host_acceptance_rate": "Métrica de solicitudes aceptadas; posterior a publicar y potencial fuga.",
        "host_is_superhost": "Estatus de reputación acumulada; un anfitrión nuevo no lo tiene al publicar.",
        "host_thumbnail_url": "Imagen/enlace de perfil; no se usará como predictor en esta fase.",
        "host_picture_url": "Imagen/enlace de perfil; no se usará como predictor en esta fase.",
        "picture_url": "Imagen del anuncio; se difiere visión por computador y no entra en el pipeline numérico.",
        "scrape_id": "Metadato del momento de recolección; no está disponible al publicar.",
        "last_scraped": "Fecha de recolección; metadato temporal, no característica del anuncio.",
        "source": "Indica cómo se encontró el anuncio en el scrape; metadato de recolección.",
        "price": "Variable objetivo; nunca debe entrar como predictor.",
        "price_quote_checkin_date": "Parte de una cotización observada; depende de una consulta posterior y no es predictor base.",
        "price_quote_checkout_date": "Parte de una cotización observada; depende de una consulta posterior y no es predictor base.",
        "price_quote_total_price": "Derivado del precio y de una cotización; fuga directa del objetivo.",
        "price_quote_price_per_night": "Derivado del precio; fuga directa del objetivo.",
        "price_quote_raw": "Respuesta cruda de una cotización; contiene información del objetivo.",
        "minimum_minimum_nights": "Resumen calculado mirando el calendario futuro; no disponible al publicar.",
        "maximum_minimum_nights": "Resumen calculado mirando el calendario futuro; no disponible al publicar.",
        "minimum_maximum_nights": "Resumen calculado mirando el calendario futuro; no disponible al publicar.",
        "maximum_maximum_nights": "Resumen calculado mirando el calendario futuro; no disponible al publicar.",
        "minimum_nights_avg_ntm": "Promedio calculado de reglas futuras; no disponible al publicar.",
        "maximum_nights_avg_ntm": "Promedio calculado de reglas futuras; no disponible al publicar.",
        "calendar_updated": "Fecha de actualización del calendario; puede ocurrir después de publicar.",
        "has_availability": "Estado observado en calendario; mezcla reservas y bloqueos posteriores.",
        "availability_30": "Disponibilidad observada a 30 días; variable futura y potencial fuga.",
        "availability_60": "Disponibilidad observada a 60 días; variable futura y potencial fuga.",
        "availability_90": "Disponibilidad observada a 90 días; variable futura y potencial fuga.",
        "availability_365": "Disponibilidad observada a 365 días; variable futura y potencial fuga.",
        "calendar_last_scraped": "Fecha de observación del calendario; metadato posterior.",
        "availability_eoy": "Disponibilidad futura observada; potencial fuga.",
        "number_of_reviews": "Reseñas acumuladas; un anfitrión nuevo no las tiene.",
        "number_of_reviews_ltm": "Reseñas de los últimos 12 meses; posterior a publicar.",
        "number_of_reviews_l30d": "Reseñas de los últimos 30 días; posterior a publicar.",
        "number_of_reviews_ly": "Reseñas del año anterior; posterior a publicar.",
        "first_review": "Fecha de la primera reseña; posterior y ausente para anfitrión nuevo.",
        "last_review": "Fecha de la última reseña; posterior y ausente para anfitrión nuevo.",
        "review_scores_rating": "Calificación acumulada; posterior y ausente para anfitrión nuevo.",
        "review_scores_accuracy": "Calificación acumulada; posterior y ausente para anfitrión nuevo.",
        "review_scores_cleanliness": "Calificación acumulada; posterior y ausente para anfitrión nuevo.",
        "review_scores_checkin": "Calificación acumulada; posterior y ausente para anfitrión nuevo.",
        "review_scores_communication": "Calificación acumulada; posterior y ausente para anfitrión nuevo.",
        "review_scores_location": "Calificación acumulada; posterior y ausente para anfitrión nuevo.",
        "review_scores_value": "Calificación acumulada; posterior y ausente para anfitrión nuevo.",
        "estimated_occupancy_l365d": "Ocupación estimada con datos posteriores y reviews; fuga.",
        "estimated_revenue_l365d": "Ingresos estimados con datos posteriores y reviews; fuga.",
        "calculated_host_listings_count": "Conteo calculado del inventario actual; puede reflejar actividad posterior.",
        "calculated_host_listings_count_entire_homes": "Conteo calculado del inventario actual; puede reflejar actividad posterior.",
        "calculated_host_listings_count_private_rooms": "Conteo calculado del inventario actual; puede reflejar actividad posterior.",
        "calculated_host_listings_count_shared_rooms": "Conteo calculado del inventario actual; puede reflejar actividad posterior.",
        "reviews_per_month": "Derivado de reviews y del tiempo de vida; fuga.",
    }
    optional = {"host_identity_verified", "host_has_profile_pic", "minimum_nights", "maximum_nights"}
    entries: list[dict[str, str]] = []
    for column in columns:
        if column in allowed:
            decision = "Permitida, opcional" if column in optional else "Permitida"
            entries.append(_policy_entry(column, decision, allowed[column]))
        elif column in deferred:
            entries.append(_policy_entry(column, "Permitida con revisión / diferida", deferred[column]))
        elif column in prohibited_exact:
            entries.append(_policy_entry(column, "Prohibida", prohibited_exact[column]))
        else:
            entries.append(
                _policy_entry(
                    column,
                    "Ambigua: no usar todavía",
                    "No hay una justificación explícita de disponibilidad temporal; revisar antes de modelar.",
                )
            )
    return entries


def render_feature_policy(columns: list[str]) -> str:
    entries = feature_policy(columns)
    lines = [
        "# Política de variables",
        "",
        "Proyecto: `data-science-airbnb-pricing` — Fase 2.",
        "",
        "La unidad de decisión es el precio que podría fijar un anfitrión nuevo antes de publicar. "
        "Por eso una variable solo puede ser predictor si existe o puede conocerse en ese momento. "
        "`id` no se usa como predictor. La política se auditó individualmente contra las columnas del "
        "snapshot de junio y no usa el snapshot de septiembre.",
        "",
        "## Categorías",
        "",
        "- `Permitida`: candidata disponible antes de publicar.",
        "- `Permitida, opcional`: disponible o potencialmente disponible antes de publicar, pero su estado puede cambiar o depende de una decisión del anfitrión; se debe evaluar en una comparación de sensibilidad.",
        "- `Permitida con revisión / diferida`: podría estar disponible, pero se excluye del pipeline actual por ambigüedad, texto o codificación pendiente.",
        "- `Prohibida`: objetivo, identificador, métrica posterior o variable con fuga.",
        "- `Ambigua: no usar todavía`: requiere una decisión adicional antes de modelar.",
        "",
        "## Decisión por columna",
        "",
        "| Columna | Decisión | Justificación |",
        "|---|---|---|",
    ]
    for entry in entries:
        reason = entry["reason"].replace("|", "\\|")
        lines.append(f"| `{entry['column']}` | {entry['decision']} | {reason} |")
    lines += [
        "",
        "## Regla de modelado",
        "",
        "El pipeline de esta fase usa solo un subconjunto numérico y categórico de variables permitidas, "
        "más `amenities_count`; `price_mxn` es el objetivo y `id` no se exporta. `neighbourhood_group_cleansed` "
        "e `instant_bookable` son conceptualmente disponibles antes de publicar, pero se excluyen del subconjunto "
        "seleccionado porque en junio tienen cobertura nula completa y quedarían como constantes. `amenities` se "
        "resume en `amenities_count`. Las demás variables diferidas o prohibidas no entran al pipeline. La decisión "
        "se toma solo con desarrollo de junio; el test de septiembre no participa.",
        "",
    ]
    return "\n".join(lines)


def _amenities_count(value: Any) -> float:
    if _is_blank(value):
        return 0.0
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return float(len(parsed))
    except (SyntaxError, ValueError):
        pass
    if text.strip("[] ") == "":
        return 0.0
    return float(len(re.findall(r"['\"]([^'\"]+)['\"]", text)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def prepare_development_dataset(
    input_path: Path = DEFAULT_JUNE_PATH,
    output_path: Path | None = DEFAULT_PROCESSED_PATH,
    cleaning_log_path: Path | None = DEFAULT_CLEANING_LOG,
    feature_policy_path: Path | None = DEFAULT_FEATURE_POLICY,
    min_category_count: int = 20,
) -> PreparedDevelopment:
    """Clean only June and fit rules on June rows.

    The returned CSV is an EDA/final-development artifact. It must not be
    created once and then split for internal validation: medians, category
    frequencies, and any future scaling or outlier thresholds must be fitted
    separately inside each training fold and applied to that fold's
    validation rows.
    """

    raw = pd.read_csv(input_path, compression="infer", low_memory=False)
    input_rows = len(raw)
    columns = list(raw.columns)
    policy = feature_policy(columns)
    raw["price_mxn"] = raw["price"].map(parse_price_mxn)
    raw["price_status"] = raw["price"].map(price_status)
    id_valid = ~raw["id"].isna() & raw["id"].astype(str).str.strip().ne("")
    valid_target = id_valid & raw["price_status"].eq("valid")

    target_status_counts = {
        status: int(raw["price_status"].eq(status).sum())
        for status in ("missing", "invalid", "zero_or_negative", "valid")
    }
    rows_removed_invalid_id = int((~id_valid).sum())
    rows_removed_invalid_price = int((id_valid & ~raw["price_status"].eq("valid")).sum())
    development = raw.loc[valid_target].copy()

    price = development["price_mxn"]
    q01, q1, median, q3, q99 = price.quantile([0.01, 0.25, 0.50, 0.75, 0.99]).tolist()
    iqr = q3 - q1
    iqr_lower = max(0.0, q1 - 1.5 * iqr)
    iqr_upper = q3 + 1.5 * iqr
    price_diagnostics = {
        "min": float(price.min()),
        "max": float(price.max()),
        "p01": float(q01),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "p99": float(q99),
        "iqr_lower": float(iqr_lower),
        "iqr_upper": float(iqr_upper),
        "below_iqr_lower_count": int((price < iqr_lower).sum()),
        "above_iqr_upper_count": int((price > iqr_upper).sum()),
        "outside_p01_p99_count": int(((price < q01) | (price > q99)).sum()),
        "extremes_removed": 0,
        "extreme_rule": "No se eliminan extremos en Fase 2; se reportan flags IQR y p01/p99 calculados solo en junio.",
        "future_sensitivity_plan": {
            "primary": "Conservar todos los precios válidos y evaluar el candidato log(precio).",
            "alternative": "Comparar, sin borrar observaciones por defecto, una especificación robusta o un tope p99 calculado dentro de cada fold de entrenamiento.",
            "validation_target": "Calcular métricas sobre el precio original de la partición de validación, no sobre un objetivo recortado.",
            "test_prohibited": True,
        },
    }

    clean = pd.DataFrame(index=development.index)
    clean["price_mxn"] = development["price_mxn"].astype(float)
    imputation_log: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        if column not in development:
            continue
        values = pd.to_numeric(development[column], errors="coerce")
        missing_before = int(values.isna().sum())
        median_value = values.median()
        if pd.isna(median_value):
            median_value = 0.0
            fallback = "all_missing_fallback_zero"
        else:
            median_value = float(median_value)
            fallback = None
        clean[column] = values.fillna(median_value).astype(float)
        imputation_log[column] = {
            "type": "median_from_june_valid_price_rows",
            "missing_before": missing_before,
            "fill_value": median_value,
            "fallback": fallback,
        }

    development["amenities_count"] = development["amenities"].map(_amenities_count)
    clean["amenities_count"] = development["amenities_count"].astype(float)
    imputation_log["amenities_count"] = {
        "type": "count_items_in_amenities_string",
        "missing_treated_as": 0,
        "fill_value": None,
    }

    category_log: dict[str, Any] = {}
    for column in CATEGORICAL_FEATURES:
        if column not in development:
            continue
        values = development[column].astype("string").str.strip().fillna("Missing").replace("", "Missing")
        counts = values.value_counts(dropna=False)
        rare_values = counts[counts < min_category_count].index.astype(str).tolist()
        values = values.where(~values.isin(rare_values), "Other")
        clean[column] = values.astype("string")
        category_log[column] = {
            "missing_after_fill": int((values == "Missing").sum()),
            "min_category_count": min_category_count,
            "rare_category_count": len(rare_values),
            "rare_categories": rare_values,
            "rare_categories_replaced_with": "Other",
        }

    clean = clean.reset_index(drop=True)
    cleaning_log: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": JUNE_SNAPSHOT_DATE,
        "input_path": _relative(input_path),
        "output_path": _relative(output_path) if output_path is not None else None,
        "uses_test_snapshot": False,
        "operations": [
            {"step": "read_june", "rows_in": input_rows, "rows_out": input_rows},
            {"step": "parse_price_to_price_mxn", "rows_in": input_rows, "rows_out": input_rows},
            {
                "step": "filter_modeling_target",
                "rows_in": input_rows,
                "rows_out": len(development),
                "rows_removed_invalid_id": rows_removed_invalid_id,
                "rows_removed_invalid_price": rows_removed_invalid_price,
                "price_status_counts": target_status_counts,
            },
            {"step": "impute_numeric_features", "rows_in": len(development), "rows_out": len(development)},
            {"step": "fill_and_group_categories", "rows_in": len(development), "rows_out": len(development)},
        ],
        "rows": {
            "input": input_rows,
            "after_valid_id_and_price": len(development),
            "output": len(clean),
            "rows_removed_total": input_rows - len(clean),
        },
        "price_status_counts": target_status_counts,
        "price_diagnostics_development_only": price_diagnostics,
        "fit_scope": "Estadísticas ajustadas sobre todo junio; artefacto para EDA y desarrollo final, no para validación interna posterior.",
        "future_validation_protocol": {
            "split_before_fit": True,
            "fit_imputation_on_training_fold_only": True,
            "fit_rare_category_rules_on_training_fold_only": True,
            "fit_scaling_on_training_fold_only": True,
            "fit_outlier_thresholds_on_training_fold_only": True,
            "apply_fitted_rules_to_validation_fold": True,
            "never_use_reserved_test": True,
        },
        "numeric_imputation": imputation_log,
        "categorical_rules": category_log,
        "min_category_count": min_category_count,
        "feature_columns": list(clean.columns),
        "id_used_as_predictor": False,
        "test_rows_saved": False,
    }
    cleaning_log = _json_safe(cleaning_log)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(output_path, index=False)
    if cleaning_log_path is not None:
        cleaning_log_path.parent.mkdir(parents=True, exist_ok=True)
        cleaning_log_path.write_text(json.dumps(cleaning_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if feature_policy_path is not None:
        feature_policy_path.parent.mkdir(parents=True, exist_ok=True)
        feature_policy_path.write_text(render_feature_policy(columns), encoding="utf-8")
    return PreparedDevelopment(data=clean, cleaning_log=cleaning_log, feature_policy=policy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--june", type=Path, default=DEFAULT_JUNE_PATH)
    parser.add_argument("--september", type=Path, default=DEFAULT_SEPTEMBER_PATH)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--cleaning-log", type=Path, default=DEFAULT_CLEANING_LOG)
    parser.add_argument("--feature-policy", type=Path, default=DEFAULT_FEATURE_POLICY)
    parser.add_argument("--min-category-count", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = [args.june, args.september, args.processed, args.split_manifest, args.cleaning_log, args.feature_policy]
    paths = [path if path.is_absolute() else REPO_ROOT / path for path in paths]
    june, september, processed, split_manifest, cleaning_log, policy = paths
    split = build_temporal_split_manifest(june, september, split_manifest)
    prepared = prepare_development_dataset(
        june,
        output_path=processed,
        cleaning_log_path=cleaning_log,
        feature_policy_path=policy,
        min_category_count=args.min_category_count,
    )
    summary = {
        "split": split,
        "cleaning_rows": prepared.cleaning_log["rows"],
        "feature_columns": prepared.cleaning_log["feature_columns"],
        "price_diagnostics_development_only": prepared.cleaning_log["price_diagnostics_development_only"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
