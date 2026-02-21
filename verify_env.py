
import sys
import os

# Add the user site-packages path exactly as the main script does
sys.path.append(os.path.expanduser("~\\AppData\\Roaming\\Python\\Python313\\site-packages"))

print(f"Python Executable: {sys.executable}")
print("Attempting imports...")

try:
    import pandas as pd
    print(f"SUCCESS: pandas version {pd.__version__}")
except ImportError as e:
    print(f"FAILURE: pandas not found - {e}")

try:
    import requests
    print(f"SUCCESS: requests version {requests.__version__}")
except ImportError as e:
    print(f"FAILURE: requests not found - {e}")

try:
    import torchaudio
    print(f"SUCCESS: torchaudio version {torchaudio.__version__}")
except ImportError as e:
    print(f"FAILURE: torchaudio not found - {e}")

try:
    import torch
    print(f"SUCCESS: torch version {torch.__version__}")
except ImportError as e:
    print(f"FAILURE: torch not found - {e}")

print("\nIf you see SUCCESS above, your environment is correctly set up.")
print("The red lines in your IDE are likely because the IDE is not checking the 'AppData' path during static analysis.")
