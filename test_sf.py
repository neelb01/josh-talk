
import os
import soundfile as sf
import torch
import numpy as np

os.makedirs("data/segments", exist_ok=True)
path = "data/segments/test_write.wav"

try:
    print(f"Testing soundfile write to: {path}")
    # Create simple sine wave
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    data = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32) # float32

    sf.write(path, data, sr, format='WAV', subtype='PCM_16')
    print("Write success with format='WAV', subtype='PCM_16'")
except Exception as e:
    print(f"Write failed: {e}")

try:
    print(f"Testing soundfile write (default format) to: {path}")
    sf.write(path, data, sr)
    print("Write success (default format)")
except Exception as e:
    print(f"Write failed: {e}")

import sys
print(f"Soundfile version: {sf.__version__}")
print(f"Libsndfile version: {sf.__libsndfile_version__}")
