import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils import detect_language, translate_to_english, estimate_cefr_ilr

# Page config
st.set_page_config(page_title="Multilingual Text Analyzer", layout="centered")
st.title("Multilingual Text Analyzer")
st.caption("Developed by Dr. Tabine")

# Load summarization model
model_id = "philschmid/bart-large-cnn-samsum"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
summarization_model = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)

# QA model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)

summarizer = HuggingFacePipeline(pipeline=summarization_model)
qa_llm = HuggingFacePipeline(pipeline=qa_model)

summary_prompt = PromptTemplate.from_template("Summarize this:\n{text}")
qa_prompt = PromptTemplate.from_template("Answer the question:\nContext: {context}\nQuestion: {question}")

summary_chain = LLMChain(llm=summarizer, prompt=summary_prompt)
qa_chain = LLMChain(llm=qa_llm, prompt=qa_prompt)

# Streamlit input
text = st.text_area("Enter your text (any language):", height=250)
analyze = st.button("Analyze")

if analyze and text:
    src_lang = detect_language(text)
    st.markdown(f"**Detected Language:** {src_lang.upper()}")

    english_text = translate_to_english(text, src_lang)
    st.markdown("**Translated Text:**")
    st.write(english_text)

    # Summary
    summary = summary_chain.run(text=english_text)
    st.markdown("**Main Idea Summary:**")
    st.write(summary)

    # CEFR/ILR Level + Justification
    level, justification = estimate_cefr_ilr(english_text)
    st.markdown(f"**Estimated CEFR / ILR Level:** {level}")
    st.markdown(f"**Justification:** {justification}")

    # Vocabulary Table
    from textblob import TextBlob
    blob = TextBlob(english_text)
    words = list(set(blob.words.lower()))
    table_data = [{"Word": word, "Translation": translate_to_english(word, "en")} for word in words[:15]]
    st.markdown("**Main Vocabulary with English Translation:**")
    st.table(table_data)

    # QA Interaction
    st.markdown("### Ask a question about the text:")
    question = st.text_input("Type your question here:")
    if question:
        answer = qa_chain.run(context=summary, question=question)
        st.markdown("**Answer:**")
        st.write(answer)
