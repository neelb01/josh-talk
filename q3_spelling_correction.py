import pandas as pd
import sys
import os
# Add user site-packages to path
sys.path.append(os.path.expanduser("~\\AppData\\Roaming\\Python\\Python313\\site-packages"))

import requests

# Configuration
SHEET5 = "sheet5.csv"
OUTPUT_CSV = "spelling_correction.csv"
DICT_URL = "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/hi_IN/hi_IN.dic"
DICT_FILE = "hi_IN.dic"

def download_dictionary():
    if os.path.exists(DICT_FILE):
        return True
    print(f"Downloading dictionary from {DICT_URL}...")
    try:
        response = requests.get(DICT_URL)
        response.raise_for_status()
        with open(DICT_FILE, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to download dictionary: {e}")
        return False

def load_dictionary(dict_path):
    words = set()
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            first_line = True
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # First line might be count
                if first_line and line.isdigit():
                    first_line = False
                    continue
                
                # word/affix
                parts = line.split('/')
                word = parts[0]
                words.add(word)
    except Exception as e:
        print(f"Error reading dictionary: {e}")
    return words

def process_spelling():
    if not download_dictionary():
        return

    valid_words = load_dictionary(DICT_FILE)
    print(f"Loaded {len(valid_words)} words from dictionary.")

    if not os.path.exists(SHEET5):
        print(f"{SHEET5} not found. Creating dummy for verification.")
        # Create dummy data if file missing
        dummy_df = pd.DataFrame({'word': ['नमस्ते', 'कम्पुटर', 'सही', 'गलत्त']})
        dummy_df.to_csv(SHEET5, index=False)
    
    # Load dataset
    try:
        df = pd.read_csv(SHEET5)
    except Exception as e:
        print(f"Error reading {SHEET5}: {e}")
        return
        
    if df.empty:
        print("Dataset is empty")
        return

    # Assume the first column contains the words
    word_col = df.columns[0]
    
    results = []
    
    for idx, row in df.iterrows():
        word = str(row[word_col]).strip()
        
        # Simple Logic: Check if in dictionary
        # Also could use heuristic rules
        is_correct = word in valid_words
        
        status = "correct spelling" if is_correct else "incorrect spelling"
        
        results.append({
            'word': word,
            'status': status
        })
        
    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    # Calculate stats
    correct_count = sum(1 for r in results if r['status'] == "correct spelling")
    print(f"Processed {len(results)} words.")
    print(f"Correct: {correct_count}")
    print(f"Incorrect: {len(results) - correct_count}")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_spelling()
