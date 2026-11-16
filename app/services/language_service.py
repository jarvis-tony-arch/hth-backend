import re
import time
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hinglish': 'Hinglish',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ta': 'Tamil',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'ur': 'Urdu',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'ar': 'Arabic',
    'zh-cn': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'pt': 'Portuguese',
    'ru': 'Russian',
}

HINGLISH_MARKERS = [
    r'\b(mera|meri|mujhe|humara|aapka)\b',
    r'\b(nhi|nahi|nahin|nai)\b',
    r'\b(hua|hui|hoga|hogi|gaya|gaye)\b',
    r'\b(baar|karo|chahiye|wapas|vapas)\b',
    r'\b(aaya|aya|mila)\b',
    r'\b(paise|rupaye|paisa)\b',
    r'\b(abhi|jaldi|turant)\b',
    r'\b(galat|sahi|kharab|toota)\b',
]


def detect_language(text: str) -> str:
    """Detect language — including Hinglish."""
    try:
        if len(text.strip()) < 3:
            return 'en'

        # Check Hinglish first (Roman script Hindi)
        text_lower = text.lower()
        hinglish_matches = sum(
            1 for p in HINGLISH_MARKERS
            if re.search(p, text_lower, re.IGNORECASE)
        )
        if hinglish_matches >= 2:
            return 'hinglish'

        lang = detect(text)
        return lang
    except LangDetectException:
        return 'en'


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate any language to English."""
    if source_lang == 'en':
        return text
    if source_lang == 'hinglish':
        # Hinglish handled by nlp_service normalize_hinglish
        # Return as-is here — NLP will normalize it
        return text
    
    try:
        time.sleep(1) # Small delay to avoid rate limits
        translated = GoogleTranslator(source=source_lang, target='en').translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"deep_translator error (to english): {e}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English text to target language."""
    if target_lang in ('en', 'hinglish') or not text:
        return text
        
    try:
        time.sleep(0.5)
        translated = GoogleTranslator(source='en', target=target_lang).translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"deep_translator error (from english): {e}")
        return text


def translate_list(items: list, target_lang: str) -> list:
    """Translate a list of strings."""
    if target_lang in ('en', 'hinglish'):
        return items
    results = []
    for item in items:
        translated = translate_from_english(item, target_lang)
        results.append(translated)
    return results


def translate_dict_values(d: dict, target_lang: str) -> dict:
    """Translate dictionary values."""
    if target_lang in ('en', 'hinglish'):
        return d
    return {
        k: translate_from_english(str(v), target_lang)
        for k, v in d.items()
    }


def get_language_name(lang_code: str) -> str:
    """Get human readable language name."""
    return SUPPORTED_LANGUAGES.get(lang_code, 'Unknown')