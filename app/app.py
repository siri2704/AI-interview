import streamlit as st
import json
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load knowledge base
with open('./kb.json', 'r', encoding='utf-8') as f:
    kb = json.load(f)

# Load sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract keywords from text
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]
    tagged = pos_tag(filtered)
    keywords = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('VB')]
    return keywords

# Select a question based on keywords
def find_related_question(kb_pool, keywords, asked_questions):
    for qna in kb_pool:
        if qna['question'] in asked_questions:
            continue
        for keyword in keywords:
            if keyword.lower() in qna['question'].lower():
                return qna
    remaining = [q for q in kb_pool if q['question'] not in asked_questions]
    return random.choice(remaining) if remaining else None

# Streamlit app setup
st.set_page_config(page_title="AI Mentor", layout="wide")
st.title("🤖 AI Mentor Chatbot")

# Initialize session state
if "state" not in st.session_state:
    st.session_state.state = "intro"
    st.session_state.asked = []
    st.session_state.qna = None
    st.session_state.user_intro = ""
    st.session_state.user_interest = ""
    st.session_state.interest_keywords = []
    st.session_state.user_answer = None
    st.session_state.show_result = False
    st.session_state.total_score = 0.0
    st.session_state.questions_answered = 0

# Introduction screen
if st.session_state.state == "intro":
    st.subheader("👋 Tell us a bit about you")
    name = st.text_input("Your name or intro:")
    interest = st.text_input("Your area of interest:")
    if st.button("Start Chat"):
        if name and interest:
            st.session_state.user_intro = name
            st.session_state.user_interest = interest
            st.session_state.interest_keywords = extract_keywords(interest)
            st.session_state.qna = find_related_question(kb, st.session_state.interest_keywords, st.session_state.asked)
            st.session_state.state = "chat"
        else:
            st.warning("Please fill both fields!")

# Chat screen
elif st.session_state.state == "chat":
    st.markdown(f"👤 **{st.session_state.user_intro}**, let's discuss **{st.session_state.user_interest}**.")

    current_qna = st.session_state.qna

    if not st.session_state.show_result:
        user_answer = st.text_input(f"🤖 {current_qna['question']}")
        if user_answer:
            if user_answer.lower() in ["exit", "quit"]:
                st.session_state.state = "end"
                st.rerun()

            st.session_state.user_answer = user_answer
            st.session_state.asked.append(current_qna['question'])

            # Compute similarity
            user_emb = model.encode(user_answer)
            correct_emb = model.encode(current_qna['answer'])
            score = cosine_similarity([user_emb], [correct_emb])[0][0]

            st.session_state.similarity_score = score
            st.session_state.show_result = True

            # Accumulate total score and count
            st.session_state.total_score += score
            st.session_state.questions_answered += 1

            st.rerun()

    else:
        st.markdown(f"### ✅ Similarity Score: **{st.session_state.similarity_score:.2f}**")

        if st.session_state.similarity_score > 0.7:
            st.success("Great answer!")
        elif st.session_state.similarity_score > 0.4:
            st.warning("Decent try! Here's a better explanation:")
        else:
            st.error("Not quite! Here's the correct answer:")

        st.markdown(f"💡 **Correct Answer:** {current_qna['answer']}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➡️ Next Question"):
                next_keywords = extract_keywords(current_qna['answer'])
                next_q = find_related_question(kb, next_keywords, st.session_state.asked)
                if next_q:
                    st.session_state.qna = next_q
                    st.session_state.user_answer = None
                    st.session_state.show_result = False
                    st.rerun()
                else:
                    st.session_state.state = "end"
                    st.rerun()

        with col2:
            if st.button("❌ Exit Chat"):
                st.session_state.state = "end"
                st.rerun()

# End screen
elif st.session_state.state == "end":
    st.balloons()
    # Calculate final score display
    total = st.session_state.total_score
    count = st.session_state.questions_answered
    max_score = count * 1.0  # max similarity per question is 1.0
    score_display = f"{total:.2f} / {max_score:.2f}"

    st.success(f"✅ You've completed the session. Your total score is: {score_display}")

    # Provide feedback based on score percentage
    if count == 0:
        st.info("No questions were answered.")
    else:
        percentage = total / max_score
        if percentage >= 0.7:
            st.success("You are doing good! Keep it up!")
        elif percentage >= 0.4:
            st.warning("Decent effort! Some improvement needed.")
        else:
            st.error("You need improvement. Keep practicing!")

    if st.button("🔁 Restart"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
