from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from huggingface_hub import login
from dotenv import load_dotenv
import os

from utils import detect_language, translate_to_english, estimate_cefr_ilr

# Load token from .env file
load_dotenv()
hf_token = os.getenv("HF_API_KEY")
if hf_token:
    login(token=hf_token)

text = input("Enter text (any language): ")

src_lang = detect_language(text)
print(f"\n[Detected Language]: {src_lang.upper()}")

translated = translate_to_english(text, src_lang)
print("\n[Translated to English]:")
print(translated)

# ✅ Safe model for summarization
summarizer = HuggingFacePipeline(
    pipeline=pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
)
summary_chain = LLMChain(llm=summarizer, prompt=PromptTemplate.from_template("Summarize:\n{text}"))
summary = summary_chain.run(text=translated)
print("\n[Summary]")
print(summary)

level, justification = estimate_cefr_ilr(translated)
print(f"\n[Estimated CEFR / ILR Level]: {level}")
print(f"[Justification]: {justification}")

qa_model = HuggingFacePipeline(
    pipeline=pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
)
qa_chain = LLMChain(llm=qa_model, prompt=PromptTemplate.from_template("Answer:\nContext: {context}\nQuestion: {question}"))

question = input("\nAsk a question about the text: ")
if question:
    answer = qa_chain.run(context=summary, question=question)
    print("\n[Answer]")
    print(answer)
