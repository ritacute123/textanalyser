import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils import detect_language, translate_to_english, estimate_cefr_ilr

# Set up Streamlit
st.set_page_config(page_title="Multilingual Text Analyzer", layout="centered")
st.title("Multilingual Text Analyzer")
st.caption("Developed by Dr. Tabine")

text = st.text_area("Enter your text (any language):", height=250)

if st.button("Analyze") and text:
    src_lang = detect_language(text)
    st.markdown(f"**Detected Language:** {src_lang.upper()}")

    english_text = translate_to_english(text, src_lang)
    st.markdown("**Translated Text:**")
    st.write(english_text)

    # Summarization setup
    model_id = "philschmid/bart-large-cnn-samsum"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

    # Run summary
    summary = summarizer(english_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    st.markdown("**Summary (Main Idea):**")
    st.write(summary)

    # Estimate level
    level, justification = estimate_cefr_ilr(english_text)
    st.markdown(f"**Estimated CEFR / ILR Level:** {level}")
    st.markdown(f"**Justification:** {justification}")

    # Extract vocab
    blob_words = pd.Series(english_text.split())
    top_words = blob_words.value_counts().head(10).reset_index()
    top_words.columns = ["Term", "Frequency"]
    top_words["English Translation"] = top_words["Term"]  # Placeholder for actual translations

    st.markdown("**Top Vocabulary with English Translation:**")
    st.dataframe(top_words)

    # QA setup
    qa_model_id = "deepset/roberta-base-squad2"
    qa_pipeline = pipeline("question-answering", model=qa_model_id, tokenizer=qa_model_id, device=-1)

    st.markdown("---")
    st.markdown("### Ask a question about the text:")
    question = st.text_input("Your Question")
    if question:
        answer = qa_pipeline(question=question, context=english_text)
        st.markdown("**Answer:**")
        st.write(answer['answer'])
