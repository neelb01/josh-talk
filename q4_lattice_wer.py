import pandas as pd
import sys
import os
# Add user site-packages to path
sys.path.append(os.path.expanduser("~\\AppData\\Roaming\\Python\\Python313\\site-packages"))

import numpy as np
import jiwer
from collections import Counter

# Configuration
SHEET6 = "sheet6.csv"
OUTPUT_REPORT = "wer_report_q4.csv"

def simple_alignment(refs, hyps):
    # This is a simplified placeholder for complex lattice alignment
    # In a real ROVER system, you align multiple sequences using dynamic programming
    # Here we will just align by index if lengths match, or use basic voting
    pass

def rover_voting(hypotheses):
    # ROVER (Recognizer Output Voting Error Reduction)
    # 1. Align hypotheses
    # 2. Vote at each position
    
    # Simple word-level voting for demonstration
    # Assumes hypotheses are roughly aligned (same number of words)
    # Real implementation needs minimizing edit distance graph
    
    maxlen = max(len(h.split()) for h in hypotheses)
    split_hyps = [h.split() for h in hypotheses]
    
    consensus = []
    for i in range(maxlen):
        words_at_i = []
        for h in split_hyps:
            if i < len(h):
                words_at_i.append(h[i])
            else:
                words_at_i.append("<eps>") # Deletion/Padding
        
        # Majority vote
        counts = Counter(words_at_i)
        best_word, count = counts.most_common(1)[0]
        
        if best_word != "<eps>":
            consensus.append(best_word)
            
    return " ".join(consensus)

def calculate_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)

def process_lattice():
    # Load data
    if pd.io.common.file_exists(SHEET6):
        df = pd.read_csv(SHEET6)
    else:
        print(f"{SHEET6} not found. Using dummy data.")
        df = pd.DataFrame({
            'reference': ["hello world this is a test"],
            'model1': ["hello world this is test"],
            'model2': ["hello word this is a test"],
            'model3': ["hello world this is a text"],
            'model4': ["hallo world this is a test"],
            'model5': ["hello world this is a test"]
        })

    results = []
    
    model_cols = [c for c in df.columns if 'model' in c or 'hyp' in c]
    ref_col = 'reference'
    
    for idx, row in df.iterrows():
        ref = str(row[ref_col])
        hyps = [str(row[c]) for c in model_cols]
        
        # 1. Individual WERs
        wers = {col: calculate_wer(ref, str(row[col])) for col in model_cols}
        
        # 2. Lattice/Consensus Construction
        consensus = rover_voting(hyps)
        
        # 3. Lattice WER
        lattice_wer = calculate_wer(ref, consensus)
        
        res = {
            'id': idx,
            'reference': ref,
            'consensus': consensus,
            'lattice_wer': lattice_wer
        }
        res.update(wers)
        results.append(res)
        
    # Validation logic
    # "Design an approach... to handle insertions, deletions..."
    # The simple voting handles this via <eps> if aligned properly.
    
    out_df = pd.DataFrame(results)
    print(out_df.describe())
    out_df.to_csv(OUTPUT_REPORT, index=False)
    print(f"Saved Q4 report to {OUTPUT_REPORT}")

if __name__ == "__main__":
    process_lattice()
