import os
import json
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# Download NLTK tokenizer model
nltk.download('punkt')

# File to keep track of processed files
PROCESSED_FILE_LOG = r"processed_files.txt"  # Using raw string for file path

# --- Extract top keywords using TF-IDF ---
def extract_keywords(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = np.array(tfidf_matrix.toarray()).flatten()
    keywords = np.array(vectorizer.get_feature_names_out())[scores.argsort()[::-1]]
    return keywords[:top_n]

# --- Load already processed file names ---
def load_processed_files():
    if os.path.exists(PROCESSED_FILE_LOG):
        with open(PROCESSED_FILE_LOG, 'r') as f:
            return set(line.strip() for line in f)
    return set()

# --- Save newly processed file names ---
def save_processed_files(processed_files):
    # Check and ensure that the directory for the log file exists
    log_dir = os.path.dirname(PROCESSED_FILE_LOG)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    with open(PROCESSED_FILE_LOG, 'a') as f:
        for file in processed_files:
            f.write(file + '\n')

# --- Read new .txt files from a folder ---
def read_text_files(folder_path, processed_files):
    data = []
    new_processed = []
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return data, new_processed
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and filename not in processed_files:
            topic = os.path.splitext(filename)[0]
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='latin1') as f:
                    content = f.read()
            data.append((topic, content))
            new_processed.append(filename)
    return data, new_processed

# --- Improvised function to generate knowledge base ---
def generate_kb(folder_path, processed_files):
    files, new_processed = read_text_files(folder_path, processed_files)
    kb = []

    for topic, content in files:
        subtopics = extract_keywords(content, top_n=10)
        sentences = sent_tokenize(content)

        for subtopic in subtopics:
            matched_sentences = [s for s in sentences if subtopic.lower() in s.lower()]
            if not matched_sentences:
                continue

            # Remove bullet placeholder artifacts
            matched_sentences = [re.sub(r'•\s*', '', s).strip() for s in matched_sentences]

            # Improved heuristic categorization of answers
            definition = matched_sentences[0] if len(matched_sentences) > 0 else f"{subtopic.capitalize()} is a key concept related to {topic.lower()}."
            example = next((s for s in matched_sentences if "example" in s.lower() or "e.g." in s.lower()), f"No specific example found for {subtopic}.")
            causes = next((s for s in matched_sentences if "cause" in s.lower() or "reason" in s.lower()), f"No clear cause found for {subtopic}.")
            prevention = next((s for s in matched_sentences if "prevent" in s.lower() or "avoid" in s.lower()), f"No known prevention strategies found for {subtopic}.")

            # Fix singular/plural question grammar
            article = "an" if re.match(r'^[aeiouAEIOU]', subtopic) else "a"
            question = f"What is {article} {subtopic.lower()}?" if subtopic else f"What is the topic?"

            kb.append({
                "topic": topic,
                "subtopic": subtopic,
                "question": question,
                "answer": {
                    "definition": definition,
                    "example": example,
                    "causes": causes,
                    "prevention": prevention
                }
            })

    return kb, new_processed

# --- Save KB as structured JSON ---
def save_kb_to_json(kb, output_file=r"C:\AI-Mentor\kb\knowledge_base1.json"):  # Using raw string for file path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load existing KB if available
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_kb = json.load(f)
    else:
        existing_kb = []

    combined_kb = existing_kb + kb

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_kb, f, indent=4, ensure_ascii=False)

    print(f"[✔] {len(kb)} new entries added to {output_file}")

# --- Run the script ---
if __name__ == "__main__":
    folder_path = r"C:\AI-Mentor\dataset_gfg_dbms"  # Raw string for folder path
    processed_files = load_processed_files()

    kb, newly_processed = generate_kb(folder_path, processed_files)

    if kb:
        save_kb_to_json(kb)
        save_processed_files(newly_processed)
    else:
        print("No new files to process.")
