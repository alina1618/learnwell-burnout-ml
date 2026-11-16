from pathlib import Path

# Корінь проєкту
BASE_DIR = Path(__file__).resolve().parent.parent

# База даних
DB_PATH = BASE_DIR / "data" / "students.db"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"                       # натренована модель
MODEL_INFO_PATH = MODELS_DIR / "model_info.json"            # версія, ознаки, метрики
PREPROC_PATH = MODELS_DIR / "preprocessing_params.json"     # медіани, моди, межі викидів
SCALER_PATH = MODELS_DIR / "my_standard_scaler.pkl"         # StandardScaler

RAW_TABLE = "student_impact"               # початкові дані
FEATURES_TABLE = "student_impact_encoded"  # очищені та закодовані ознаки, на них вчили модель у лабі 3

# Службові колонки
ID_COL = "Student_ID"
TIME_COL = "timestamp"
TARGET_COL = "Burnout_Risk_Level"

CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}
HIGH_CODE = 2

# Частка нових даних, які відкладаємо від тренування
NEW_INPUT_SHARE = 0.10

# Найкращі гіперпараметри RandomizedSearchCV
BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 4,
    "random_state": 42,
}