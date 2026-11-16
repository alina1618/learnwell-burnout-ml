import json
from datetime import datetime

import joblib
import pandas as pd
from fastapi import APIRouter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

from app.config import (BEST_PARAMS, CLASS_NAMES, FEATURES_TABLE, HIGH_CODE, ID_COL,
                        MODEL_INFO_PATH, MODEL_PATH, MODELS_DIR, NEW_INPUT_SHARE,
                        RAW_TABLE, TARGET_COL, TIME_COL)
from app.database import create_tables, get_connection
from app.preprocessing import fit_params

router = APIRouter()


def calc_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "recall_high": round(recall_score(y_true, y_pred, labels=[HIGH_CODE], average="macro"), 4),
    }


@router.post("/train-model", tags=["1. Тренування"],
             summary="Натренувати модель і записати передбачення на train у БД")

def train_model():
    create_tables()
    conn = get_connection()

    # Всі записи з таблиці ознак
    df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE}", conn)
    df = df.dropna(subset=[TIME_COL])                    # запис без timestamp не можна впорядкувати
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # 90:10 за часом
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n_train = int(len(df) * (1 - NEW_INPUT_SHARE))
    train = df.iloc[:n_train]
    new_input = df.iloc[n_train:]

    feature_columns = [c for c in df.columns if c not in (ID_COL, TIME_COL, TARGET_COL)]
    X_train, y_train = train[feature_columns], train[TARGET_COL]
    X_new, y_new = new_input[feature_columns], new_input[TARGET_COL]

    # Тренування моделі з найкращими параметрами
    model = RandomForestClassifier(**BEST_PARAMS, n_jobs=-1)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    metrics = {
        "train": calc_metrics(y_train, train_pred),
        "new_input": calc_metrics(y_new, model.predict(X_new)),
    }

    # Збереження моделі і всього, що потрібно для передбачення
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    # Важливість ознак
    importance = (pd.Series(model.feature_importances_, index=feature_columns)
                  .sort_values(ascending=False).round(4))
    importance.to_csv(MODELS_DIR / "feature_importance.csv", header=["importance"])

    raw = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", conn)
    fit_params(raw[raw[ID_COL].isin(train[ID_COL])])

    info = {"model_version": version, "feature_columns": feature_columns,
            "params": BEST_PARAMS, "metrics": metrics,
            "n_train": len(train), "n_new_input": len(new_input)}
    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # Передбачення на train - таблиця predictions
    proba = model.predict_proba(X_train)
    now = datetime.now().isoformat(timespec="seconds")
    train_log = pd.DataFrame({
        "Student_ID": train[ID_COL].values,
        "model_version": version,
        "source": "train",
        "predicted_code": train_pred,
        "predicted_label": [CLASS_NAMES[c] for c in train_pred],
        "proba_low": proba[:, 0].round(4),
        "proba_medium": proba[:, 1].round(4),
        "proba_high": proba[:, 2].round(4),
        "true_label": [CLASS_NAMES[c] for c in y_train],
        "timestamp": train[TIME_COL].astype(str).values,
        "created_at": now,
    })
    train_log.to_sql("predictions", conn, if_exists="append", index=False)

    split = pd.DataFrame({"Student_ID": df[ID_COL],
                          "split": ["train"] * len(train) + ["new_input"] * len(new_input),
                          "model_version": version})
    split.to_sql("data_split", conn, if_exists="replace", index=False)

    # Метрики - у таблицю model_metrics
    pd.DataFrame([
        {"model_name": "Random Forest (API)", "stage": f"deployed {version}", "source": src,
         **m, "best_params": str(BEST_PARAMS)}
        for src, m in metrics.items()
    ]).to_sql("model_metrics", conn, if_exists="append", index=False)

    conn.close()
    return {"status": "модель натреновано", "model_version": version,
            "train_rows": len(train), "new_input_rows": len(new_input),
            "metrics": metrics, "top_5_features": importance.head(5).to_dict(),
            "saved_to": str(MODEL_PATH.name)}