from fastapi import FastAPI

from app.inference import router as inference_router
from app.monitoring import router as monitoring_router
from app.training import router as training_router

app = FastAPI(
    title="LearnWell — прогноз ризику вигорання",
    description="Лабораторна робота 4, варіант 2 (Impact of AI on Students). "
                "Спочатку виконайте **1. Тренування**, потім **2. Передбачення**, далі **3. Моніторинг** (ЛР5).",
    version="1.0",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,   # сховати блок Schemas внизу сторінки
        "tryItOutEnabled": True,          # поля одразу можна редагувати, без кнопки Try it out
        "displayRequestDuration": True,   # показувати, скільки тривав запит
    },
)

app.include_router(training_router)    # POST /train-model
app.include_router(inference_router)   # POST /predict
app.include_router(monitoring_router)  # GET /monitor (ЛР5)


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Сервер працює. Документація: /docs"}
