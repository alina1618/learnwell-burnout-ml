import json

import joblib
import numpy as np
import pandas as pd

from app.config import PREPROC_PATH, SCALER_PATH

# Ordinal encoding
YEAR_MAP = {"Freshman": 0, "Sophomore": 1, "Junior": 2, "Senior": 3, "Graduate": 4}

# One-Hot encoding (k-1)
NOMINAL_COLUMNS = ["Major_Category", "Institutional_Policy",
                   "Primary_Use_Case", "Prompt_Engineering_Skill"]

# Колонки, у яких у  обрізали викиди за правилом IQR
CAP_COLUMNS = ["Weekly_GenAI_Hours", "Pre_Semester_GPA", "Traditional_Study_Hours",
               "Perceived_AI_Dependency", "Post_Semester_GPA", "Skill_Retention_Score"]


def fit_params(raw_train: pd.DataFrame) -> dict:
    """Рахує всі значення для обробки на тренувальних даних і зберігає їх у json."""
    post = raw_train["Post_Semester_GPA"]
    q1, q3 = post.quantile(0.25), post.quantile(0.75)

    params = {
        "median_pre_gpa": float(raw_train["Pre_Semester_GPA"].median()),
        "median_anxiety": float(raw_train["Anxiety_Level_During_Exams"].median()),
        "median_tool_diversity": float(raw_train["Tool_Diversity"].median()),
        "post_gpa_extreme_value": float(q3 + 3 * (q3 - q1)),  # значення з кінця розподілу
        "mode_use_case": str(raw_train["Primary_Use_Case"].mode()[0]),
        "cap_bounds": {},
    }
    for col in CAP_COLUMNS:
        c_q1, c_q3 = raw_train[col].quantile(0.25), raw_train[col].quantile(0.75)
        iqr = c_q3 - c_q1
        params["cap_bounds"][col] = [float(c_q1 - 1.5 * iqr), float(c_q3 + 1.5 * iqr)]

    with open(PREPROC_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    return params


def preprocess(raw: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Сирі дані студентів -> таблиця ознак у тому ж вигляді, що student_impact_encoded."""
    with open(PREPROC_PATH, encoding="utf-8") as f:
        params = json.load(f)
    scaler = joblib.load(SCALER_PATH)

    df = raw.copy()

    # Пропуски 
    df["Pre_Semester_GPA"] = df["Pre_Semester_GPA"].fillna(params["median_pre_gpa"])
    df["Post_Semester_GPA"] = df["Post_Semester_GPA"].fillna(params["post_gpa_extreme_value"])
    df["Anxiety_Level_During_Exams_is_NA"] = np.where(df["Anxiety_Level_During_Exams"].isnull(), 1, 0)
    df["Anxiety_Level_During_Exams"] = df["Anxiety_Level_During_Exams"].fillna(params["median_anxiety"])
    df["Primary_Use_Case"] = df["Primary_Use_Case"].fillna(params["mode_use_case"])
    df["Prompt_Engineering_Skill"] = df["Prompt_Engineering_Skill"].fillna("Missing")
    df["Tool_Diversity"] = df["Tool_Diversity"].fillna(params["median_tool_diversity"])

    # Аномалії
    for col, (low, high) in params["cap_bounds"].items():
        df[col] = df[col].clip(lower=low, upper=high)

    # Кодування
    df["Year_of_Study"] = df["Year_of_Study"].map(YEAR_MAP)
    df["Paid_Subscription"] = df["Paid_Subscription"].astype(int)
    df = pd.get_dummies(df, columns=NOMINAL_COLUMNS, dtype=int)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Масштабування збереженим scaler
    scaled_cols = list(scaler.feature_names_in_)
    df[scaled_cols] = scaler.transform(df[scaled_cols])
    return df