from langdetect import detect
from deep_translator import GoogleTranslator

def detect_language(text):
    """Detect the language of the input text"""
    return detect(text)

def translate_to_english(text, src_lang):
    """Translate the input text to English"""
    translator = GoogleTranslator(source=src_lang, target='en')
    return translator.translate(text)

def estimate_cefr_ilr(text):
    """Estimate the ILR level based on the input text"""
    # Example logic for estimation. Replace with actual logic for level estimation.
    level = "ILR Level 2"
    justification = "The text contains basic vocabulary and simple sentence structures."
    return level, justification
