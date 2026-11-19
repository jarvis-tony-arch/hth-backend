from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .database import Base

class TicketDB(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String, unique=True, index=True)
    raw_query = Column(String)
    ai_title = Column(String)
    ai_summary = Column(String)
    urgency = Column(String)
    sentiment = Column(String)
    confidence = Column(Float)
    status = Column(String, default="NEW")
    issues = Column(String) # Stored as comma-separated
    departments = Column(String) # Stored as comma-separated
    created_at = Column(DateTime, default=datetime.utcnow)
