from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob
from utils import detect_language, translate_to_english, estimate_cefr_ilr

text = input("Enter your text in any language:\n")

# 1. Detect & translate
lang = detect_language(text)
translated = translate_to_english(text, lang)
print(f"\n[Detected Language]: {lang.upper()}")
print(f"\n[Translated Text]:\n{translated}")

# 2. Summarize
model_id = "philschmid/bart-large-cnn-samsum"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
summary = summarizer(translated, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
print(f"\n[Summary]:\n{summary}")

# 3. Estimate CEFR / ILR level
level, justification = estimate_cefr_ilr(translated)
print(f"\n[CEFR / ILR Level]: {level}")
print(f"[Justification]: {justification}")

# 4. Vocabulary table
print("\n[Main Vocabulary with English Translation]:")
blob = TextBlob(translated)
vocab = sorted(set(blob.words.lower()))[:10]
for word in vocab:
    try:
        translation = translate_to_english(word, "en")
    except:
        translation = "N/A"
    print(f"- {word} → {translation}")

# 5. Interactive QA
qa = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
while True:
    question = input("\nAsk a question about the text (or press Enter to exit): ")
    if not question:
        break
    answer = qa(question=question, context=summary)
    print("Answer:", answer['answer'])
