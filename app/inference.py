import json
import uuid
from datetime import datetime

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.config import CLASS_NAMES, ID_COL, MODEL_INFO_PATH, MODEL_PATH, RAW_TABLE, TARGET_COL
from app.database import create_tables, get_connection
from app.preprocessing import preprocess
from app.schemas import PredictRequest

router = APIRouter()

RAW_COLUMNS = ["Student_ID", "Major_Category", "Year_of_Study", "Pre_Semester_GPA",
               "Weekly_GenAI_Hours", "Primary_Use_Case", "Prompt_Engineering_Skill",
               "Tool_Diversity", "Paid_Subscription", "Traditional_Study_Hours",
               "Perceived_AI_Dependency", "Institutional_Policy", "Anxiety_Level_During_Exams",
               "Post_Semester_GPA", "Skill_Retention_Score", "timestamp"]

# Модель в пам'яті
_cache = {"mtime": None, "model": None, "info": None}


def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(400, "Модель ще не натренована: спочатку виконайте POST /train-model")
    mtime = MODEL_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["model"] = joblib.load(MODEL_PATH)
        with open(MODEL_INFO_PATH, encoding="utf-8") as f:
            _cache["info"] = json.load(f)
        _cache["mtime"] = mtime
    return _cache["model"], _cache["info"]


def get_input_data(request: PredictRequest, conn) -> tuple[pd.DataFrame, str]:
    """Повертає сирі дані студентів і звідки вони прийшли ('json' або 'db')."""
    if request.students:
        rows = [s.model_dump() for s in request.students]
        return pd.DataFrame(rows), "json"

    if request.student_ids:
        marks = ",".join("?" * len(request.student_ids))
        df = pd.read_sql(f"SELECT * FROM {RAW_TABLE} WHERE Student_ID IN ({marks})",
                         conn, params=request.student_ids)
        if df.empty:
            raise HTTPException(404, "Студентів з такими Student_ID немає в БД")
        return df, "db"

    if request.new_input:
        query = f"""SELECT r.* FROM {RAW_TABLE} r
                    JOIN data_split s ON r.Student_ID = s.Student_ID
                    WHERE s.split = 'new_input'"""
        if request.limit:
            query += f" LIMIT {request.limit}"
        return pd.read_sql(query, conn), "db"

    raise HTTPException(422, "Передайте students, student_ids або new_input = true")


@router.post("/predict", tags=["2. Передбачення"],
             summary="Спрогнозувати рівень ризику вигорання (Low / Medium / High)")
def predict(request: PredictRequest):
    create_tables()
    model, info = load_model()                              # 1. модель з диска
    conn = get_connection()

    raw, input_source = get_input_data(request, conn)       # 2. вхідні дані
    raw = raw.reset_index(drop=True)

    X = preprocess(raw, info["feature_columns"])            # 3. обробка як у лабі 2
    codes = model.predict(X)                                # 4. передбачення
    proba = model.predict_proba(X)

    request_id = uuid.uuid4().hex[:8]
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.cursor()
    results = []

    for i in range(len(raw)):
        row = raw.loc[i]

        # Вхідні параметри 
        values = {col: (None if pd.isna(row.get(col)) else row.get(col)) for col in RAW_COLUMNS}
        values["Paid_Subscription"] = int(values["Paid_Subscription"])
        values = {k: (v.item() if hasattr(v, "item") else v) for k, v in values.items()}
        cur.execute(
            f"""INSERT INTO inference_inputs (request_id, input_source, {", ".join(RAW_COLUMNS)}, created_at)
                VALUES (?, ?, {", ".join("?" * len(RAW_COLUMNS))}, ?)""",
            [request_id, input_source, *values.values(), now],
        )
        input_id = cur.lastrowid

        # Результат -> predictions
        code = int(codes[i])
        true_label = row.get(TARGET_COL) if TARGET_COL in raw.columns else None
        cur.execute(
            """INSERT INTO predictions (Student_ID, model_version, source, predicted_code,
                   predicted_label, proba_low, proba_medium, proba_high, true_label,
                   timestamp, inference_input_id, created_at)
               VALUES (?, ?, 'inference', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [values[ID_COL], info["model_version"], code, CLASS_NAMES[code],
             round(float(proba[i, 0]), 4), round(float(proba[i, 1]), 4), round(float(proba[i, 2]), 4),
             true_label, values["timestamp"], input_id, now],
        )
        results.append({"inference_input_id": input_id, "Student_ID": values[ID_COL],
                        "predicted_label": CLASS_NAMES[code],
                        "proba_high": round(float(proba[i, 2]), 4),
                        "true_label": true_label})

    conn.commit()
    conn.close()
    return {"request_id": request_id, "model_version": info["model_version"],
            "input_source": input_source, "count": len(results),
            "predictions": results[:50]}   # у відповіді максимум 50, у БД — усі