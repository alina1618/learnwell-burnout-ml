# LearnWell — прогноз ризику емоційного вигорання студентів

![CI](https://github.com/alina1618/learnwell-burnout-ml/actions/workflows/ci.yml/badge.svg)

Навчальний проєкт з курсу «Основи смарт-технологій і систем» (ЛР1–ЛР5).
Повний ML-цикл: підготовка даних → модель Random Forest → розгортання як FastAPI-сервіс
з логуванням у SQLite → моніторинг дрейфу даних і якості моделі через Evidently AI.

Сервіс прогнозує рівень ризику емоційного вигорання студента (`Burnout_Risk_Level`: **Low / Medium / High**)
за його навчальними та поведінковими характеристиками (години роботи з GenAI, самостійне навчання,
тривожність перед іспитами, GPA тощо). Модель тренується і робить передбачення лише через HTTP-запити,
а всі дії записуються в базу даних `data/students.db`.

**Стек:** Python · pandas · scikit-learn · FastAPI · Pydantic · SQLite · Evidently AI · pytest · GitHub Actions

---

## Етапи проєкту

| Лабораторна | Що зроблено | Де в проєкті |
| --- | --- | --- |
| ЛР1 | Завантаження датасету з Kaggle в SQLite, додано `timestamp` | `src/LR_1.ipynb`, таблиця `student_impact` |
| ЛР2 | EDA, обробка пропусків і викидів, кодування, масштабування | `src/LR_2.ipynb`, `app/preprocessing.py` |
| ЛР3 | Порівняння моделей, підбір гіперпараметрів Random Forest | `src/LR_3.ipynb`, таблиця `model_metrics` |
| ЛР4 | Розгортання моделі: `/train-model`, `/predict`, логування в БД | `app/` |
| ЛР5 | Моніторинг: Data Drift, Target Drift, Classification, Data Quality | `app/monitoring.py`, `src/LR_5.ipynb`, `src/monitor.py` |

---

## Структура проєкту

```
learnwell-burnout-ml/
├── .github/workflows/ci.yml   # CI: автоматичний запуск тестів на кожен push
├── app/
│   ├── __init__.py
│   ├── config.py              # шляхи, назви таблиць, гіперпараметри з ЛР3
│   ├── database.py            # підключення до БД, створення predictions та inference_inputs
│   ├── preprocessing.py       # обробка нових даних так само, як у ЛР2
│   ├── training.py            # POST /train-model
│   ├── schemas.py             # формат і перевірка JSON-запиту
│   ├── inference.py           # POST /predict
│   ├── monitoring.py          # GET /monitor, GET /monitor/summary (ЛР5)
│   └── main.py                # точка входу FastAPI
├── data/
│   └── students.db            # база даних з ЛР1–ЛР3
├── models/
│   └── my_standard_scaler.pkl # StandardScaler з ЛР2 (решта файлів створюється після /train-model)
├── reports/                   # HTML-дашборди Evidently
├── src/
│   ├── LR_1.ipynb … LR_5.ipynb  # ноутбуки лабораторних
│   ├── monitor.py             # скрипт автоматичного моніторингу
│   ├── show_db.py             # перегляд таблиць БД
│   └── reset_demo.py          # скидання до стану «до тренування»
├── tests/
│   └── test_api.py            # тести API
├── docs/
│   └── project_description.md
├── pytest.ini
├── requirements.txt
└── README.md
```

Після `/train-model` у `models/` з'являються `model.pkl`, `model_info.json` (версія, ознаки, метрики),
`preprocessing_params.json` (медіани, мода, межі викидів, пораховані на train) і `feature_importance.csv`.

---

## Швидкий старт (Windows, PowerShell)

```powershell
# 1. створити та активувати віртуальне середовище
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. встановити бібліотеки
pip install -r requirements.txt

# 3. запустити сервер (з кореня проєкту)
python -m uvicorn app.main:app --reload
```

> Якщо PowerShell не дає активувати середовище:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Після запуску:
- сервер: http://127.0.0.1:8000
- інтерактивна документація (Swagger UI): http://127.0.0.1:8000/docs

Перед першим запуском у `data/` має лежати `students.db`, а в `models/` — `my_standard_scaler.pkl`.

**Порядок роботи:** `POST /train-model` → `POST /predict` → `GET /monitor/summary`.

---

## Ендпоінти

| Метод | Шлях | Що робить |
| --- | --- | --- |
| POST | `/train-model` | тренує модель, зберігає її в `models/`, логує прогнози з `source='train'` |
| POST | `/predict` | прогноз для JSON-студентів або студентів з БД, логує з `source='inference'` |
| GET | `/monitor?report=...` | будує HTML-дашборд Evidently (`data_drift`, `target_drift`, `classification`, `data_quality`) |
| GET | `/monitor/summary` | короткий підсумок моніторингу в JSON |

### `POST /train-model`

Тіло запиту не потрібне.

```bash
curl -X POST http://127.0.0.1:8000/train-model
```

Що робить ендпоінт:
1. читає всі записи з таблиці ознак `student_impact_encoded`;
2. сортує їх за `timestamp` і ділить у співвідношенні 90:10 на `train` (44 117) та `new_input` (4 902);
3. тренує Random Forest з найкращими параметрами з ЛР3
   (`n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=4`);
4. зберігає модель у `models/model.pkl`, а параметри обробки — у `models/preprocessing_params.json`;
5. записує передбачення на train-даних у таблицю `predictions` з `source = 'train'`,
   а метрики — у таблицю `model_metrics`.

Приклад відповіді:

```json
{
  "status": "модель натреновано",
  "model_version": "20261025_144700",
  "train_rows": 44117,
  "new_input_rows": 4902,
  "metrics": {
    "train":     {"accuracy": 0.773, "f1_macro": 0.7653, "recall_high": 0.6214},
    "new_input": {"accuracy": 0.532, "f1_macro": 0.5294, "recall_high": 0.4556}
  },
  "saved_to": "model.pkl"
}
```

### `POST /predict`

Ендпоінт приймає один із трьох варіантів тіла запиту.

**1. Дані студента в JSON** (значення як у початковій таблиці `student_impact`; кодування й
масштабування сервер робить сам):

```json
{
  "students": [{
    "Major_Category": "STEM",
    "Year_of_Study": "Junior",
    "Pre_Semester_GPA": 3.2,
    "Weekly_GenAI_Hours": 22.5,
    "Primary_Use_Case": "Direct_Answer_Generation",
    "Prompt_Engineering_Skill": "Beginner",
    "Tool_Diversity": 2,
    "Paid_Subscription": true,
    "Traditional_Study_Hours": 5.0,
    "Perceived_AI_Dependency": 7,
    "Institutional_Policy": "Allowed_With_Citation",
    "Anxiety_Level_During_Exams": 8,
    "Post_Semester_GPA": 2.9,
    "Skill_Retention_Score": 55.0
  }]
}
```

Поля `Pre_Semester_GPA`, `Post_Semester_GPA`, `Anxiety_Level_During_Exams`, `Primary_Use_Case`,
`Prompt_Engineering_Skill`, `Tool_Diversity` можна не передавати — пропуски заповняться так само, як у ЛР2.
Неправильні значення (наприклад, `"Major_Category": "Law"` або від'ємні години) повертають помилку `422`.

**2. Студенти з БД за Student_ID:**

```json
{"student_ids": [100001, 100002, 100003]}
```

**3. Усі відкладені 10% записів (new_input) з БД:**

```json
{"new_input": true}
```

Приклад відповіді:

```json
{
  "request_id": "4be753b0",
  "model_version": "20261025_144700",
  "input_source": "db",
  "count": 3,
  "predictions": [
    {"inference_input_id": 1, "Student_ID": 100001, "predicted_label": "High",   "proba_high": 0.7215, "true_label": "High"},
    {"inference_input_id": 2, "Student_ID": 100002, "predicted_label": "Low",    "proba_high": 0.0854, "true_label": "Low"},
    {"inference_input_id": 3, "Student_ID": 100003, "predicted_label": "Medium", "proba_high": 0.4501, "true_label": "Medium"}
  ]
}
```

Кожен виклик зберігає вхідні дані в `inference_inputs`, а результат — у `predictions` з `source = 'inference'`.

---

## Моніторинг моделі (ЛР5)

Моніторинг порівнює два набори даних за допомогою **Evidently AI**:
- **reference** — студенти, на яких тренувалась модель (`predictions`, `source='train'`);
- **current** — запити до API (`inference_inputs` + `predictions`, `source='inference'`).

| Звіт | Що перевіряє |
| --- | --- |
| `data_drift.html` | чи змінились розподіли 14 вхідних ознак (Wasserstein для чисел, Jensen-Shannon для категорій, поріг 0,1) |
| `target_drift.html` | чи змінились частки класів у справжніх мітках і в прогнозах |
| `classification.html` | якість моделі на нових даних: accuracy, F1, recall, матриця помилок |
| `data_quality.html` | пропуски, діапазони, унікальні значення |
| `simulated_drift.html` | перевірка, що моніторинг ловить штучно змодельований дрейф |

Три способи запуску:
1. **Ноутбук** `src/LR_5.ipynb` — повний аналіз з усіма звітами.
2. **Скрипт** — зберігає звіти з датою в назві (історія запусків), попереджає, якщо дрейфує ≥ 50% ознак:
   ```powershell
   python src\monitor.py
   ```
3. **API** — `GET /monitor/summary` (JSON) і `GET /monitor?report=data_drift` (HTML-дашборд, відкривати в новій вкладці).

Приклад відповіді `/monitor/summary`:

```json
{
  "reference_rows": 44117,
  "current_rows": 4905,
  "drifted_features": [],
  "share_drifted": 0.0,
  "current_quality": {"accuracy": 0.536, "f1_macro": 0.533, "recall_high": 0.45},
  "need_retraining": false
}
```

---

## Результати

| Набір | Accuracy | F1-macro | Recall High |
| --- | --- | --- | --- |
| train | 0.773 | 0.765 | 0.621 |
| new_input (нові дані) | 0.532 | 0.529 | 0.456 |

Найважливіші ознаки: `Weekly_GenAI_Hours` (0.276), `Perceived_AI_Dependency` (0.109),
`Traditional_Study_Hours` (0.091), `Pre_Semester_GPA` (0.087), `Prompt_Engineering_Skill` (0.084).

Висновки моніторингу:
- дрейфу даних немає: 0 з 14 ознак, максимальна оцінка ≈ 0.018 при порозі 0.1;
- дрейфу цілі немає: частки класів Low/Medium/High практично однакові;
- якість на нових даних на рівні тестової вибірки ЛР3, тож деградації немає;
- під час симуляції «сесії» (години GenAI ×2, тривожність +2) Evidently виявив дрейф саме у змінених ознаках,
  а частка прогнозів High зросла з 17% до 38%.

### Обмеження

- Модель недооцінює клас **High** (recall 0.45, 17% прогнозів High при 25% насправді).
  Наступний крок — `class_weight`, підбір порогу для High, інші моделі (gradient boosting).
- `timestamp` у ЛР1 згенеровано випадково, тому поділ за часом не дає реального дрейфу.
- `Post_Semester_GPA` і `Skill_Retention_Score` відомі лише після семестру. Для прогнозу під час
  семестру їх варто виключити й перевірити, як зміниться якість.

---

## Структура БД

| Таблиця | Звідки | Що зберігає |
| --- | --- | --- |
| `student_impact` | ЛР1 | початкові дані з Kaggle + `timestamp` (50 000 записів) |
| `student_impact_encoded` | ЛР2 | очищені, закодовані й масштабовані ознаки (49 019 записів), зв'язок 1:1 за `Student_ID` |
| `model_metrics` | ЛР3 / ЛР4 | метрики моделей; після кожного `/train-model` додаються рядки `Random Forest (API)` |
| `predictions` | ЛР4 | передбачення моделі, `source = 'train'` або `'inference'` |
| `inference_inputs` | ЛР4 | вхідні параметри кожного запиту до `/predict` |
| `data_split` | ЛР4 | які студенти потрапили в `train`, а які — в `new_input` |

**predictions:** `id`, `Student_ID`, `model_version`, `source`, `predicted_code`, `predicted_label`,
`proba_low`, `proba_medium`, `proba_high`, `true_label`, `timestamp`, `inference_input_id`, `created_at`.

**inference_inputs:** `id`, `request_id`, `input_source`, `Student_ID`, 14 ознак студента
(`Major_Category` … `Skill_Retention_Score`), `timestamp`, `created_at`.

Зв'язок: `predictions.inference_input_id → inference_inputs.id`.

SQL-запит для перегляду прогнозів разом із вхідними даними:

```sql
SELECT p.Student_ID, p.predicted_label, p.proba_high, p.true_label,
       i.input_source, i.Weekly_GenAI_Hours, i.Anxiety_Level_During_Exams
FROM predictions p
JOIN inference_inputs i ON p.inference_input_id = i.id
WHERE p.source = 'inference'
LIMIT 5;
```

Швидкий перегляд усіх таблиць: `python src\show_db.py`.

---

## Тести і CI

Тести API лежать у `tests/test_api.py`: сервер запускається, Swagger доступний,
невалідні запити відхиляються з кодом `422`.

```powershell
python -m pytest -v
```

На кожен push GitHub Actions автоматично встановлює залежності й запускає тести
(`.github/workflows/ci.yml`). Статус видно на значку CI вгорі сторінки та у вкладці **Actions**.