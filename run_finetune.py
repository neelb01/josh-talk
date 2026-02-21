import subprocess
import sys

result = subprocess.run(
    [sys.executable, "q1_finetune_whisper.py"],
    capture_output=True, text=True, timeout=600
)
print("=== STDOUT ===")
print(result.stdout)
print("=== STDERR ===")
print(result.stderr)
print(f"=== EXIT CODE: {result.returncode} ===")
