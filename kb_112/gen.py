import os
import re
import csv

# Directory containing your GFG .txt files
INPUT_DIR = r"C:\AI-Mentor\dataset_gfg_dbms"
OUTPUT_CSV = r"C:\AI-Mentor\kb_112\gfg_knowledge_base.json"

# Regex patterns to identify questions
QUESTION_PATTERNS = [
    re.compile(r"^(?:Q\d+\.|Q\d+:|Q\.|Q:|Q\d+|Q)\s*(.+\?)", re.IGNORECASE),
    re.compile(r"^(?:\d+\.\s*)(.+\?)"),
    re.compile(r"^#+\s*(Q\d*\.?\s*)?(.+\?)"),  # Markdown headings
    re.compile(r"^(.+\?)$"),  # Fallback: any line ending with '?'
]

# Regex to identify answer headings
ANSWER_PATTERNS = [
    re.compile(r"^(A\d+\.|A\d+:|A\.|A:|Answer:|Ans:|Ans\.)\s*", re.IGNORECASE),
    re.compile(r"^#+\s*(A\d*\.?\s*)?(.+)", re.IGNORECASE),
]

def is_question(line):
    for pattern in QUESTION_PATTERNS:
        m = pattern.match(line.strip())
        if m:
            # Return the actual question part
            return m.group(1).strip() if m.groups() else line.strip()
    return None

def is_answer_heading(line):
    for pattern in ANSWER_PATTERNS:
        if pattern.match(line.strip()):
            return True
    return False

def clean_line(line):
    # Remove markdown, extra spaces, bullets, etc.
    line = re.sub(r"^#+\s*", "", line)
    line = line.strip(" *-\n\r\t")
    return line

def extract_qa_from_file(filepath):
    qa_pairs = []
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    question = None
    answer_lines = []
    in_answer = False

    for idx, line in enumerate(lines):
        line_clean = clean_line(line)
        if not line_clean:
            continue

        # Detect question
        q_text = is_question(line_clean)
        if q_text:
            # Save previous Q&A if present
            if question and answer_lines:
                qa_pairs.append((question, " ".join(answer_lines)))
            question = q_text
            answer_lines = []
            in_answer = True
            continue

        # Detect answer heading
        if is_answer_heading(line_clean):
            in_answer = True
            continue

        # If in answer, collect lines
        if in_answer and question:
            # Stop collecting answer if a new question starts
            if is_question(line_clean):
                qa_pairs.append((question, " ".join(answer_lines)))
                question = is_question(line_clean)
                answer_lines = []
            else:
                answer_lines.append(line_clean)

    # Add last Q&A
    if question and answer_lines:
        qa_pairs.append((question, " ".join(answer_lines)))

    return qa_pairs

def main():
    all_qa = []
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(INPUT_DIR, fname)
        qa_pairs = extract_qa_from_file(fpath)
        for q, a in qa_pairs:
            all_qa.append({"Question": q, "Answer": a, "Source": fname})

    # Write to CSV
    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["Question", "Answer", "Source"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_qa:
            writer.writerow(row)
    print(f"Extracted {len(all_qa)} Q&A pairs to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
