from fastapi import APIRouter
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.nlp_service import analyze
from app.services.language_service import (
    detect_language,
    translate_to_english,
    translate_from_english,
    translate_list,
    get_language_name,
)

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_query(request: AnalyzeRequest):
    original_text = request.text

    # Step 1 — Detect language
    detected_lang = request.language or detect_language(original_text)

    # Step 2 — Translate to English for NLP
    english_text = translate_to_english(original_text, detected_lang)

    # Step 3 — Run NLP on English text
    result = analyze(english_text)

    # Step 4 — Translate results back if not English
    if detected_lang != 'en':
        # Translate issue displays and departments
        for issue in result['issues']:
            issue['display_translated'] = translate_from_english(
                issue['display'], detected_lang
            )
            issue['department_translated'] = translate_from_english(
                issue['department'], detected_lang
            )

        # Translate XAI reasons
        result['xai_reasons'] = translate_list(
            result['xai_reasons'], detected_lang
        )

        # Translate follow-up questions
        result['follow_up_questions'] = translate_list(
            result['follow_up_questions'], detected_lang
        )

        # Translate AI summary and title
        result['ai_summary'] = translate_from_english(
            result['ai_summary'], detected_lang
        )
        result['ai_title'] = translate_from_english(
            result['ai_title'], detected_lang
        )

        # Translate missing info
        result['missing_info'] = translate_list(
            result['missing_info'], detected_lang
        )

        # Translate urgency and sentiment labels
        result['urgency'] = result['urgency']  # keep as is (badge)
        result['sentiment'] = translate_from_english(
            result['sentiment'], detected_lang
        )

    # Step 5 — Add language metadata
    result['detected_language'] = detected_lang
    result['detected_language_name'] = get_language_name(detected_lang)
    result['original_text'] = original_text
    result['translated_text'] = english_text

    return result