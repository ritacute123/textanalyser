import streamlit as st
from transformers import pipeline
from utils import detect_language, translate_to_english, estimate_cefr_ilr

# Page config
st.set_page_config(page_title="Text Analyzer", layout="centered")
st.title("Text Analyzer")
st.caption("Developed by Dr. Tabine")

# Streamlit input
text = st.text_area("Enter your text (any language):", height=250)
analyze = st.button("Analyze")

if analyze and text:
    # Step 1: Detect language
    src_lang = detect_language(text)
    st.markdown(f"**Detected Language:** {src_lang.upper()}")

    # Step 2: Translate to English
    english_text = translate_to_english(text, src_lang)
    st.markdown("**Translated Text:**")
    st.write(english_text)

    # Step 3: Estimate ILR/CEFR Level
    level, justification = estimate_cefr_ilr(english_text)
    st.markdown(f"**Estimated ILR Level:** {level}")
    st.markdown(f"**Justification:** {justification}")
