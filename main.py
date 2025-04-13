from utils import detect_language, translate_to_english, estimate_cefr_ilr

def main():
    text = input("Enter text (any language): ")
    src_lang = detect_language(text)
    print(f"Detected Language: {src_lang.upper()}")

    english_text = translate_to_english(text, src_lang)
    print(f"Translated Text:\n{english_text}")

    # Estimation
    level, justification = estimate_cefr_ilr(english_text)
    print(f"Estimated CEFR / ILR Level: {level}")
    print(f"Justification: {justification}")
