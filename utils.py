from langdetect import detect
from deep_translator import GoogleTranslator
from textblob import TextBlob

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text, src_lang):
    if src_lang == "en":
        return text
    try:
        return GoogleTranslator(source=src_lang, target="en").translate(text)
    except:
        return text

def estimate_cefr_ilr(text):
    blob = TextBlob(text)
    word_count = len(blob.words)
    sentence_count = len(blob.sentences)

    justification = f"Detected {word_count} words and {sentence_count} sentences."

    if word_count < 50:
        level = "A1 / ILR 0+"
    elif word_count < 100:
        level = "A2 / ILR 1"
    elif word_count < 200:
        level = "B1 / ILR 2"
    elif word_count < 300:
        level = "B2 / ILR 2+"
    elif word_count < 500:
        level = "C1 / ILR 3"
    else:
        level = "C2 / ILR 4-5"

    return level, justification
