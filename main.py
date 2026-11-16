from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.analyze import router as analyze_router

app = FastAPI(
    title="HTH Support AI",
    version="2.0.0",
    description="Multilingual Smart Customer Support Intelligence"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "HTH NLP Backend",
        "version": "2.0.0",
        "features": [
            "multi-label classification",
            "multilingual support",
            "entity extraction",
            "urgency detection",
            "sentiment analysis",
            "confidence-aware routing",
            "explainable AI",
            "proactive follow-up"
        ]
    }

@app.get("/languages")
async def supported_languages():
    from app.services.language_service import SUPPORTED_LANGUAGES
    return {"supported_languages": SUPPORTED_LANGUAGES}