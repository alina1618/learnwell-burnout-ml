"""Моніторинг моделі (ЛР5): порівняння еталонних (train) і поточних (inference) даних
за допомогою Evidently AI. Використовується і в ноутбуці src/LR_5.ipynb, і в ендпоінті /monitor."""
import sqlite3
from datetime import datetime

import pandas as pd
from evidently import DataDefinition, Dataset, MulticlassClassification, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.config import BASE_DIR, DB_PATH

REPORTS_DIR = BASE_DIR / "reports"

NUM_FEATURES = ["Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity",
                "Traditional_Study_Hours", "Perceived_AI_Dependency",
                "Anxiety_Level_During_Exams", "Post_Semester_GPA", "Skill_Retention_Score"]
CAT_FEATURES = ["Major_Category", "Year_of_Study", "Primary_Use_Case",
                "Prompt_Engineering_Skill", "Paid_Subscription", "Institutional_Policy"]
FEATURES = NUM_FEATURES + CAT_FEATURES
COLUMNS = ["Student_ID", "timestamp"] + FEATURES + ["target", "prediction"]


def load_reference(conn) -> pd.DataFrame:
    """Еталон: студенти, на яких тренувалась остання модель (predictions, source='train'),
    з ознаками у звичайному вигляді з таблиці student_impact."""
    cols = ", ".join(f"f.{c}" for c in FEATURES)
    return pd.read_sql(f"""
        SELECT p.Student_ID, p.timestamp, {cols},
               p.true_label AS target, p.predicted_label AS prediction
        FROM predictions p
        JOIN student_impact f ON f.Student_ID = p.Student_ID
        WHERE p.source = 'train'
          AND p.model_version = (SELECT MAX(model_version) FROM predictions WHERE source = 'train')
    """, conn)


def load_current(conn) -> pd.DataFrame:
    """Поточні дані: усі запити до /predict (inference_inputs) разом із прогнозом
    і справжнім класом, якщо він відомий (predictions, source='inference')."""
    cols = ", ".join(f"i.{c}" for c in FEATURES)
    return pd.read_sql(f"""
        SELECT p.Student_ID, p.timestamp, {cols},
               p.true_label AS target, p.predicted_label AS prediction
        FROM predictions p
        JOIN inference_inputs i ON i.id = p.inference_input_id
        WHERE p.source = 'inference'
    """, conn)


def align(df: pd.DataFrame) -> pd.DataFrame:
    """Однакова структура і типи для обох наборів."""
    df = df[COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in NUM_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Paid_Subscription"] = df["Paid_Subscription"].map({1: "Yes", 0: "No", True: "Yes", False: "No"})
    for c in CAT_FEATURES + ["target", "prediction"]:
        df[c] = df[c].astype("object")
    return df


DATA_DEFINITION = DataDefinition(
    numerical_columns=NUM_FEATURES,
    categorical_columns=CAT_FEATURES + ["target", "prediction"],
    timestamp="timestamp",
    classification=[MulticlassClassification(target="target", prediction_labels="prediction")],
)


def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, data_definition=DATA_DEFINITION)


def drift_table(snapshot) -> pd.DataFrame:
    """Витягує з результату Evidently таблицю: ознака, метод, оцінка, поріг, чи є дрейф."""
    rows = []
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        if cfg.get("type", "").endswith("ValueDrift"):
            method, score, thr = cfg["method"], m["value"], cfg["threshold"]
            drift = score < thr if "p_value" in method.lower() else score >= thr
            rows.append({"column": cfg["column"], "method": method, "score": round(score, 4),
                         "threshold": thr, "drift": bool(drift)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def build_reports(reference: pd.DataFrame, current: pd.DataFrame, out_dir=REPORTS_DIR, suffix="") -> dict:
    """Генерує 4 HTML-звіти і повертає короткий підсумок."""
    out_dir.mkdir(exist_ok=True)
    ref_ds, cur_ds = to_dataset(reference), to_dataset(current)
    files = {}

    drift = Report([DataDriftPreset(columns=FEATURES)]).run(current_data=cur_ds, reference_data=ref_ds)
    files["data_drift"] = out_dir / f"data_drift{suffix}.html"
    drift.save_html(str(files["data_drift"]))

    target = Report([ValueDrift(column="target"), ValueDrift(column="prediction")]).run(
        current_data=cur_ds, reference_data=ref_ds)
    files["target_drift"] = out_dir / f"target_drift{suffix}.html"
    target.save_html(str(files["target_drift"]))

    labelled = current.dropna(subset=["target"])
    clf = Report([ClassificationPreset()]).run(current_data=to_dataset(labelled), reference_data=ref_ds)
    files["classification"] = out_dir / f"classification{suffix}.html"
    clf.save_html(str(files["classification"]))

    quality = Report([DataSummaryPreset()]).run(current_data=cur_ds, reference_data=ref_ds)
    files["data_quality"] = out_dir / f"data_quality{suffix}.html"
    quality.save_html(str(files["data_quality"]))

    features = drift_table(drift)
    tgt = drift_table(target)
    return {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "current_with_target": len(labelled),
        "drifted_features": features.loc[features["drift"], "column"].tolist(),
        "share_drifted": round(features["drift"].mean(), 3),
        "top_features": features.head(3)[["column", "score"]].to_dict(orient="records"),
        "target_drift": tgt.to_dict(orient="records"),
        "files": {k: v.name for k, v in files.items()},
    }


# ----------------------------------------------------------------- бонус: ендпоінт /monitor
router = APIRouter()

REPORT_BUILDERS = {
    "data_drift": lambda: Report([DataDriftPreset(columns=FEATURES)]),
    "target_drift": lambda: Report([ValueDrift(column="target"), ValueDrift(column="prediction")]),
    "classification": lambda: Report([ClassificationPreset()]),
    "data_quality": lambda: Report([DataSummaryPreset()]),
}


def load_both():
    conn = sqlite3.connect(DB_PATH)
    try:
        reference, current = align(load_reference(conn)), align(load_current(conn))
    except Exception:
        raise HTTPException(400, "Немає даних: спочатку виконайте /train-model і /predict")
    finally:
        conn.close()
    if reference.empty or current.empty:
        raise HTTPException(400, "Немає даних: спочатку виконайте /train-model і /predict")
    return reference, current


@router.get("/monitor", tags=["3. Моніторинг"], response_class=HTMLResponse,
            summary="Побудувати ОДИН звіт Evidently і показати HTML-дашборд")
def monitor(report: str = "data_drift"):
    """report: data_drift | target_drift | classification | data_quality"""
    if report not in REPORT_BUILDERS:
        raise HTTPException(422, f"report має бути одним з: {list(REPORT_BUILDERS)}")
    reference, current = load_both()
    if report == "classification":
        current = current.dropna(subset=["target"])
    snapshot = REPORT_BUILDERS[report]().run(current_data=to_dataset(current),
                                             reference_data=to_dataset(reference))
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{report}.html"
    snapshot.save_html(str(path))
    return path.read_text(encoding="utf-8")


@router.get("/monitor/summary", tags=["3. Моніторинг"],
            summary="Короткий підсумок моніторингу у JSON (без HTML, швидко)")
def monitor_summary():
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    reference, current = load_both()
    ref_ds, cur_ds = to_dataset(reference), to_dataset(current)
    features = drift_table(Report([DataDriftPreset(columns=FEATURES)]).run(
        current_data=cur_ds, reference_data=ref_ds))
    target = drift_table(Report([ValueDrift(column="target"), ValueDrift(column="prediction")]).run(
        current_data=cur_ds, reference_data=ref_ds))
    labelled = current.dropna(subset=["target"])
    quality = {
        "accuracy": round(accuracy_score(labelled.target, labelled.prediction), 4),
        "f1_macro": round(f1_score(labelled.target, labelled.prediction, average="macro"), 4),
        "recall_high": round(recall_score(labelled.target, labelled.prediction,
                                          labels=["High"], average="macro"), 4),
    } if len(labelled) else None
    share = float(features["drift"].mean())
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "current_with_target": len(labelled),
        "drifted_features": features.loc[features["drift"], "column"].tolist(),
        "share_drifted": round(share, 3),
        "top_features": features.head(3)[["column", "score"]].to_dict(orient="records"),
        "target_drift": target.to_dict(orient="records"),
        "current_quality": quality,
        "need_retraining": share >= 0.5,
    }
