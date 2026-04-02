#!/usr/bin/env python3
"""Test script to diagnose issues with CounterfactualGen"""

import sys
import subprocess
import time

# Run the main script and capture output
proc = subprocess.Popen(
    ["python3", "src/counterfactual_gen/CounterfactualGen_18k.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

start_time = time.time()
timeout = 300  # 5 minutes timeout

print("Starting script... (will monitor for 5 minutes)")
print("=" * 70)

output_lines = []
try:
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        output_lines.append(line.rstrip())
        print(line.rstrip())
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"\n[TIMEOUT after {timeout}s] Terminating...")
            proc.terminate()
            break
            
except KeyboardInterrupt:
    print("\n[Interrupted by user]")
    proc.terminate()

proc.wait()
print("=" * 70)
print(f"Exit code: {proc.returncode}")
print(f"Total lines: {len(output_lines)}")
print(f"Last 10 lines:")
for line in output_lines[-10:]:
    print(f"  {line}")
