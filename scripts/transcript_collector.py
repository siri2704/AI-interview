import os
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Configurable paths
TRANSCRIPT_FOLDER = 'datasets_transcripts_dbms'
SKIPPED_FILE = os.path.join('datasets', 'skipped_dbms.txt')  # <-- Updated
EXCEL_FILE = 'datasets/DBMS.xlsx'  # Change if needed

# Ensure directories exist
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(SKIPPED_FILE), exist_ok=True)

def load_video_ids_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df['Video ID'].dropna().unique().tolist()

def save_transcript(video_id, transcript_text):
    with open(os.path.join(TRANSCRIPT_FOLDER, f"{video_id}.txt"), 'w', encoding='utf-8') as f:
        f.write(transcript_text)

def is_already_done(video_id):
    return os.path.exists(os.path.join(TRANSCRIPT_FOLDER, f"{video_id}.txt"))

def log_skipped(video_id):
    with open(SKIPPED_FILE, 'a') as f:
        f.write(video_id + '\n')

def main():
    video_ids = load_video_ids_from_excel(EXCEL_FILE)
    print(f"🎯 Total videos found: {len(video_ids)}")

    for i, video_id in enumerate(video_ids, 1):
        print(f"\n[{i}/{len(video_ids)}] Processing: {video_id}")

        if is_already_done(video_id):
            print("✅ Already processed. Skipping.")
            continue

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = ' '.join([item['text'] for item in transcript_list])
            save_transcript(video_id, full_text)
            print("📝 Transcript saved.")
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"⚠️ Transcript not available: {e},skipping it and putting it skipped file.")
            log_skipped(video_id)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            log_skipped(video_id)

if __name__ == "__main__":
    main()
