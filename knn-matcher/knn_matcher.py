import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

# --- Download NLTK Data ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# --- Load Knowledge Base ---
with open(r'C:\AI-Mentor\kb.json', 'r', encoding='utf-8') as f:
    kb = json.load(f)

# --- Load Sentence Transformer ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Keyword Extraction ---
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]
    tagged = pos_tag(filtered)
    keywords = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('VB')]
    return keywords

# --- Short Answer Truncation ---
def get_short_answer(full_answer, limit=300):
    short = full_answer.strip().split("\n")[0][:limit]
    return short + ('...' if len(full_answer) > limit else '')

# --- Find Related Question Based on Keywords ---
def find_related_question(kb_pool, keywords, asked_questions):
    for qna in kb_pool:
        if qna['question'] in asked_questions:
            continue
        for keyword in keywords:
            if keyword.lower() in qna['question'].lower():
                return qna
    # fallback to random unseen question
    remaining = [q for q in kb_pool if q['question'] not in asked_questions]
    return random.choice(remaining) if remaining else None

# --- Main Chat Logic ---
def chat():
    print("🤖 Hi! Tell me something about yourself!")
    user_intro = input("👤 You: ")

    print("🤖 Nice to meet you! What are your areas of interest?")
    user_interest = input("👤 You: ")
    interest_keywords = extract_keywords(user_interest)

    print(f"🤖 Awesome! Let's start with some questions about {user_interest}!\n")

    asked = []
    current_qna = find_related_question(kb, interest_keywords, asked)

    while current_qna:
        current_q = current_qna['question']
        full_answer = current_qna['answer']  # ✅ USE the 'answer' field directly
        asked.append(current_q)

        print(f"🤖 {current_q}")
        user_input = input("👤 You: ")

        if user_input.lower() in ['exit', 'quit']:
            print("🤖 Goodbye! You did great today!")
            break

        # --- Similarity Check ---
        user_emb = model.encode(user_input)
        correct_emb = model.encode(full_answer)
        sim_score = cosine_similarity([user_emb], [correct_emb])[0][0]

        print(f"\n🤖 Similarity Score: {sim_score:.2f}")
        if sim_score > 0.7:
            print("🤖 Great answer! ✅")
        elif sim_score > 0.4:
            print("🤖 Decent try! Here's a better explanation:")
        else:
            print("🤖 Good attempt! Let me show you the correct idea:")

        print(f"➡️ {get_short_answer(full_answer)}\n")

        # --- Get next question ---
        next_keywords = extract_keywords(user_input)
        if not next_keywords:
            next_keywords = extract_keywords(full_answer)

        current_qna = find_related_question(kb, next_keywords, asked)

    print("🤖 We’ve covered all related questions! You did an amazing job!")

# --- Run ---
if __name__ == '__main__':
    chat()
