from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..database import get_db

router = APIRouter()

@router.post("/tickets")
def create_ticket(ticket: schemas.TicketCreate, db: Session = Depends(get_db)):
    db_ticket = models.TicketDB(**ticket.dict())
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