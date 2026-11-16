from pydantic import BaseModel
from typing import Optional

class AnalyzeRequest(BaseModel):
    text: str
    language: Optional[str] = None  # override auto-detect

class IssueResult(BaseModel):
    intent: str
    display: str
    department: str
    confidence: float
    display_translated: Optional[str] = None
    department_translated: Optional[str] = None

class AnalyzeResponse(BaseModel):
    issues: list[IssueResult]
    departments: list[str]
    urgency: str
    sentiment: str
    confidence: float
    entities: dict
    escalate: bool
    ai_title: str
    ai_summary: str
    xai_reasons: list[str]
    missing_info: list[str]
    follow_up_questions: list[str]
    # Language info
    detected_language: str
    detected_language_name: str
    original_text: str
    translated_text: str