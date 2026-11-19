from pydantic import BaseModel
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    text: str

class IssueDetail(BaseModel):
    issue: str
    department: str
    confidence: float

class AnalyzeResponse(BaseModel):
    original_text: str
    processed_text: str
    detected_language: str
    hinglish_detected: bool
    issues: List[IssueDetail]
    urgency: str
    sentiment: str
    confidence: float
    departments: List[str]
    explanations: List[str]
    follow_up_questions: List[str]

class TicketCreate(BaseModel):
    ticket_id: str
    raw_query: str
    ai_title: str
    ai_summary: str
    urgency: str
    sentiment: str
    confidence: float
    issues: List[str]
    departments: List[str]

class TicketResponse(TicketCreate):
    status: str