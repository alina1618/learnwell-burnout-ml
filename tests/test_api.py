from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

STUDENT = {
    "Major_Category": "STEM",
    "Year_of_Study": "Junior",
    "Pre_Semester_GPA": 3.2,
    "Weekly_GenAI_Hours": 22.5,
    "Primary_Use_Case": "Direct_Answer_Generation",
    "Prompt_Engineering_Skill": "Beginner",
    "Tool_Diversity": 2,
    "Paid_Subscription": True,
    "Traditional_Study_Hours": 5.0,
    "Perceived_AI_Dependency": 7,
    "Institutional_Policy": "Allowed_With_Citation",
    "Anxiety_Level_During_Exams": 8,
    "Post_Semester_GPA": 2.9,
    "Skill_Retention_Score": 55.0,
}


def test_root():
    """Сервер запускається і відповідає."""
    assert client.get("/").status_code == 200


def test_docs():
    """Swagger-документація доступна."""
    assert client.get("/docs").status_code == 200


def test_predict_invalid_category():
    """Неіснуюча спеціальність 'Law' відхиляється з кодом 422."""
    student = {**STUDENT, "Major_Category": "Law"}
    assert client.post("/predict", json={"students": [student]}).status_code == 422


def test_predict_negative_hours():
    """Від'ємні години відхиляються з кодом 422."""
    student = {**STUDENT, "Weekly_GenAI_Hours": -5}
    assert client.post("/predict", json={"students": [student]}).status_code == 422