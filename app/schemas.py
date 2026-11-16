#Опис формату JSON для /predict
from typing import Literal, Optional

from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    Student_ID: Optional[int] = None
    Major_Category: Literal["Arts", "Business", "Humanities", "Medical", "STEM"]
    Year_of_Study: Literal["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
    Pre_Semester_GPA: Optional[float] = Field(None, ge=0, le=4)
    Weekly_GenAI_Hours: float = Field(ge=0, le=168)
    Primary_Use_Case: Optional[Literal["Copywriting/Drafting", "Debugging/Troubleshooting",
                                       "Direct_Answer_Generation", "Ideation",
                                       "Summarizing_Reading"]] = None
    Prompt_Engineering_Skill: Optional[Literal["Beginner", "Intermediate", "Advanced"]] = None
    Tool_Diversity: Optional[int] = Field(None, ge=1, le=5)
    Paid_Subscription: bool
    Traditional_Study_Hours: float = Field(ge=0, le=168)
    Perceived_AI_Dependency: int = Field(ge=1, le=10)
    Institutional_Policy: Literal["Actively_Encouraged", "Allowed_With_Citation", "Strict_Ban"]
    Anxiety_Level_During_Exams: Optional[int] = Field(None, ge=1, le=10)
    Post_Semester_GPA: Optional[float] = Field(None, ge=0, le=4)
    Skill_Retention_Score: float = Field(ge=0, le=100)
    timestamp: Optional[str] = None


class PredictRequest(BaseModel):
    # Заповнюється один із трьох варіантів:
    students: Optional[list[StudentInput]] = None    # дані студентів прямо в JSON
    student_ids: Optional[list[int]] = None          # взяти студентів з БД за Student_ID
    new_input: bool = False                          # взяти з БД відкладені 10% записів
    limit: Optional[int] = Field(None, ge=1, description="скільки записів new_input брати")

    model_config = {"json_schema_extra": {"examples": [{
        "students": [{
            "Major_Category": "STEM", "Year_of_Study": "Junior",
            "Pre_Semester_GPA": 3.2, "Weekly_GenAI_Hours": 22.5,
            "Primary_Use_Case": "Direct_Answer_Generation",
            "Prompt_Engineering_Skill": "Beginner", "Tool_Diversity": 2,
            "Paid_Subscription": True, "Traditional_Study_Hours": 5.0,
            "Perceived_AI_Dependency": 7, "Institutional_Policy": "Allowed_With_Citation",
            "Anxiety_Level_During_Exams": 8, "Post_Semester_GPA": 2.9,
            "Skill_Retention_Score": 55.0,
        }]
    }]}}