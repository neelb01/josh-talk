import sys
import os
# Add user site-packages to path
sys.path.append(os.path.expanduser("~\\AppData\\Roaming\\Python\\Python313\\site-packages"))

import pandas as pd
import requests
import json
import torchaudio
import torch

# Force soundfile backend for Windows


# Create directories
DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
SEGMENTS_DIR = os.path.join(DATA_DIR, "segments")
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SEGMENTS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

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

def process_dataset(csv_path):
    df = pd.read_csv(csv_path)
    
    # Fix URLs
    df['rec_url_gcp'] = df['rec_url_gcp'].str.replace('https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/', 'https://storage.googleapis.com/upload_goai/')
    df['transcription_url_gcp'] = df['transcription_url_gcp'].str.replace('https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/', 'https://storage.googleapis.com/upload_goai/')
    
    # Store processed segments
    segments_data = []

    for idx, row in df.iterrows():
        rec_id = row['recording_id']
        audio_url = row['rec_url_gcp']
        json_url = row['transcription_url_gcp']
        
        # Paths
        audio_path = os.path.join(AUDIO_DIR, f"{rec_id}.wav")
        json_path = os.path.join(TRANSCRIPTS_DIR, f"{rec_id}.json")
        
        # Download
        print(f"Processing {rec_id}...")
        if not download_file(audio_url, audio_path):
            continue
        if not download_file(json_url, json_path):
            continue
            
        # Load JSON
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                transcription_data = json.load(f)
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
            continue

        # Load Audio
        try:
            import soundfile as sf
            wav, sr = sf.read(audio_path)
            waveform = torch.from_numpy(wav).float()
            # If stereo, convert to mono or transpose if needed (torchaudio returns [channels, time])
            # soundfile returns [time] or [time, channels]
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()
            sample_rate = sr
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            continue
            
        # Process segments
        for i, segment in enumerate(transcription_data):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '')
            
            if end_time <= start_time:
                continue
                
            # Convert time to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # Slice audio
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # Check length - minimum samples (e.g., 0.1s worth or just > 0)
            if segment_waveform.shape[1] < 160: # < 10ms at 16kHz
                continue
            
            # Save segment
            segment_filename = f"{rec_id}_{i}.wav"
            segment_path = os.path.join(SEGMENTS_DIR, segment_filename)
            
            try:
                if segment_waveform.dim() > 1 and segment_waveform.shape[0] < segment_waveform.shape[1]:
                     segment_waveform_np = segment_waveform.t().numpy()
                else:
                     segment_waveform_np = segment_waveform.numpy()
                
                sf.write(segment_path, segment_waveform_np, sample_rate, format='WAV', subtype='PCM_16')
            except Exception as e:
                print(f"Error saving segment {segment_path}: {e}")
                continue
            
            segments_data.append({
                'id': f"{rec_id}_{i}",
                'audio_path': segment_path,
                'sentence': text,
                'duration': end_time - start_time
            })
            
    # Save train.csv
    train_df = pd.DataFrame(segments_data)
    train_df.to_csv("train.csv", index=False)
    print(f"Created train.csv with {len(train_df)} segments.")

if __name__ == "__main__":
    if not os.path.exists("sheet1.csv"):
        print("sheet1.csv not found!")
    else:
        process_dataset("sheet1.csv")
