from pydantic import BaseModel
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    text: str
    language: Optional[str] = None

class IssueDetail(BaseModel):
    intent: str        # Changed from 'issue' to 'intent'
    display: str       # Added this
    department: str
    confidence: float
    matched_count: int # Added this

class AnalyzeResponse(BaseModel):
    original_text: str
    processed_text: str
    detected_language: str
    detected_language_name: str
    hinglish_detected: bool
    issues: List[IssueDetail]
    departments: List[str]
    urgency: str
    sentiment: str
    confidence: float
    xai_reasons: List[str]  # Changed from 'explanations' to 'xai_reasons'
    missing_info: List[str] # Added this
    follow_up_questions: List[str]
    translated_text: Optional[str] = None

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