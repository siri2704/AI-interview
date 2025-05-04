import os
import re
import json
from typing import List, Dict

# === CONFIGURATION ===
INPUT_FOLDER = r'C:\AI-Mentor\dataset_gfg_dbms'   # 🔁 Change this to your actual folder path
OUTPUT_FILE = r'C:\AI-Mentor\kb_111\knowledge_base.json'  # 🔁 Output path for the JSON file

# === Step 1: Load all .txt files from the folder ===
def load_txt_files(folder_path: str) -> Dict[str, str]:
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            full_path = os.path.join(folder_path, filename)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    texts[filename] = f.read()
            except Exception as e:
                print(f"[ERROR] Could not read {filename}: {e}")
    return texts

# === Step 2: Extract Q&A pairs from text ===
def extract_qas(text: str) -> List[Dict[str, str]]:
    qas = []
    # Split text by logical sections: ###, numbers, bullets
    blocks = re.split(r'(?:###|\nQ?\d+\.\s*|\n\d+\.\s*|\n-+\s*|\n•\s*)', text)
    for block in blocks:
        block = block.strip()
        if not block or len(block.split()) < 5:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        question = lines[0]
        if not question.endswith('?'):
            question = f"What is {question.strip(':-')}?"
        answer = "\n".join(lines[1:]).strip()
        if len(answer) < 20:
            continue
        qas.append({'question': question, 'answer': answer})
    return qas

# === Step 3: Build the knowledge base ===
def build_knowledge_base(folder_path: str) -> List[Dict[str, str]]:
    all_texts = load_txt_files(folder_path)
    kb = []
    for filename, content in all_texts.items():
        qas = extract_qas(content)
        for qa in qas:
            qa['source'] = filename
            kb.append(qa)
    return kb

# === Step 4: Save the knowledge base to JSON ===
def save_as_json(kb: List[Dict[str, str]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kb, f, indent=4, ensure_ascii=False)
    print(f"✅ Knowledge base saved to: {output_path}")
    print(f"✅ Total Q&A pairs: {len(kb)}")

# === Run the pipeline ===
if __name__ == "__main__":
    kb = build_knowledge_base(INPUT_FOLDER)
    save_as_json(kb, OUTPUT_FILE)
