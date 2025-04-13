import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from huggingface_hub import login
import os

from utils import detect_language, translate_to_english, estimate_cefr_ilr

# Load Hugging Face token from secrets or environment
hf_token = st.secrets.get("HF_API_KEY") or os.getenv("HF_API_KEY")
if hf_token:
    login(token=hf_token)

# Streamlit UI setup
st.set_page_config(page_title="Multilingual Text Analyzer", layout="centered")
st.title("Multilingual Text Analyzer")
st.caption("Developed by Dr. Tabine")

# ✅ Manually load summarization model (Cloud-safe)
model_id = "philschmid/bart-large-cnn-samsum"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
summarization_model = pipeline("summarization", model=model, tokenizer=tokenizer)

# QA model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)

# LangChain pipelines
summarizer = HuggingFacePipeline(pipeline=summarization_model)
qa_llm = HuggingFacePipeline(pipeline=qa_model)

# Prompt templates
summary_prompt = PromptTemplate.from_template("Summarize this:\n{text}")
qa_prompt = PromptTemplate.from_template("Answer the question:\nContext: {context}\nQuestion: {question}")

summary_chain = LLMChain(llm=summarizer, prompt=summary_prompt)
qa_chain = LLMChain(llm=qa_llm, prompt=qa_prompt)

# Streamlit text input
text = st.text_area("Enter your text (any language):", height=250)

if text:
    src_lang = detect_language(text)
    st.markdown(f"**Detected Language:** {src_lang.upper()}")

    english_text = translate_to_english(text, src_lang)
    st.markdown("**Translated Text:**")
    st.write(english_text)

    summary = summary_chain.run(text=english_text)
    st.markdown("**Summary:**")
    st.write(summary)

    level, justification = estimate_cefr_ilr(english_text)
    st.markdown(f"**Estimated CEFR / ILR Level:** {level}")
    st.markdown(f"**Justification:** {justification}")

    question = st.text_input("Ask a question about the text:")
    if question:
        answer = qa_chain.run(context=summary, question=question)
        st.markdown("**Answer:**")
        st.write(answer)
