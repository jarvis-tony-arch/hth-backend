from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db

router = APIRouter()

@router.post("/tickets")
def create_ticket(ticket: schemas.TicketCreate, db: Session = Depends(get_db)):
    # Check if ticket already exists to prevent unique constraint errors
    existing = db.query(models.TicketDB).filter(models.TicketDB.ticket_id == ticket.ticket_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Ticket ID already exists")

    # Safely convert lists to strings
    issues_str = ",".join(ticket.issues) if ticket.issues else ""
    depts_str = ",".join(ticket.departments) if ticket.departments else ""
    
    db_ticket = models.TicketDB(
        ticket_id=ticket.ticket_id,
        raw_query=ticket.raw_query,
        ai_title=ticket.ai_title,
        ai_summary=ticket.ai_summary,
        urgency=ticket.urgency,
        sentiment=ticket.sentiment,
        confidence=ticket.confidence,
        issues=issues_str,
        departments=depts_str
    )
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)
    return db_ticket

@router.get("/tickets")
def get_tickets(db: Session = Depends(get_db)):
    return db.query(models.TicketDB).order_by(models.TicketDB.created_at.desc()).all()

@router.put("/tickets/{ticket_id}")
def update_ticket_status(ticket_id: str, status: str, db: Session = Depends(get_db)):
    db_ticket = db.query(models.TicketDB).filter(models.TicketDB.ticket_id == ticket_id).first()
    if not db_ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    db_ticket.status = status
    db.commit()
    return db_ticket