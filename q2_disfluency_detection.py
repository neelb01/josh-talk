import pandas as pd
import sys
import os
# Add user site-packages to path
sys.path.append(os.path.expanduser("~\\AppData\\Roaming\\Python\\Python313\\site-packages"))

import json
import torchaudio
import requests

# Force soundfile backend for Windows
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

# Directories
DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
CLIPS_DIR = os.path.join(DATA_DIR, "disfluency_clips")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")

os.makedirs(CLIPS_DIR, exist_ok=True)

# File paths
SHEET1 = "sheet1.csv"
SHEET3 = "sheet3.csv"
OUTPUT_CSV = "disfluencies.csv"

def load_disfluencies(csv_path):
    df = pd.read_csv(csv_path)
    # Flatten all columns into a single list of unique words/phrases
    disfluencies = set()
    for col in df.columns:
        for val in df[col].dropna().astype(str):
            val = val.strip()
            if val:
                disfluencies.add(val)
    return disfluencies

def download_file(url, filepath):
    if os.path.exists(filepath):
        return True
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def process_disfluencies():
    if not os.path.exists(SHEET1) or not os.path.exists(SHEET3):
        print(f"Missing input CSVs: {SHEET1} or {SHEET3}")
        return

    # Load targets
    targets = load_disfluencies(SHEET3)
    print(f"Loaded {len(targets)} unique disfluency targets.")

    # Load dataset
    df = pd.read_csv(SHEET1)
    
    # Fix URLs
    df['rec_url_gcp'] = df['rec_url_gcp'].str.replace('https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/', 'https://storage.googleapis.com/upload_goai/')
    df['transcription_url_gcp'] = df['transcription_url_gcp'].str.replace('https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/', 'https://storage.googleapis.com/upload_goai/')
    
    results = []
    
    for idx, row in df.iterrows():
        rec_id = row['recording_id']
        json_url = row['transcription_url_gcp']
        audio_url = row['rec_url_gcp']
        
        # Paths
        json_path = os.path.join(TRANSCRIPTS_DIR, f"{rec_id}.json")
        audio_path = os.path.join(AUDIO_DIR, f"{rec_id}.wav")
        
        # Ensure JSON exists (download if needed)
        if not os.path.exists(json_path):
            if not download_file(json_url, json_path):
                continue
                
        # Load JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
        except Exception as e:
            print(f"Error loading JSON {json_path}: {e}")
            continue
            
        # Check text for disfluencies
        segments_with_disfluency = []
        for i, seg in enumerate(segments):
            text = seg.get('text', '')
            found = []
            for t in targets:
                # Simple substring check (could be improved with tokenization)
                if t in text:
                    found.append(t)
            
            if found:
                segments_with_disfluency.append((i, seg, found))
        
        if not segments_with_disfluency:
            continue
            
        print(f"Found {len(segments_with_disfluency)} disfluencies in {rec_id}")
        
        # Download Audio if needed
        if not os.path.exists(audio_path):
            print(f"Downloading audio for {rec_id}...")
            if not download_file(audio_url, audio_path):
                continue
        
        # Load Audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            continue
            
        # Clip and Save
        for i, seg, found_types in segments_with_disfluency:
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '')
            
            if end <= start:
                continue
                
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            clip = waveform[:, start_sample:end_sample]
            clip_name = f"{rec_id}_seg{i}_disfluency.wav"
            clip_path = os.path.join(CLIPS_DIR, clip_name)
            
            torchaudio.save(clip_path, clip, sample_rate)
            
            # Add to results
            results.append({
                'disfluency_type': ", ".join(found_types),
                'audio_segment_url': clip_path, # Local path for now
                'start_time (s)': start,
                'end_time (s)': end,
                'transcription_snippet': text,
                'notes': f"Found {found_types} in segment {i}"
            })

    # Save CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Disfluency detection complete. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_disfluencies()
