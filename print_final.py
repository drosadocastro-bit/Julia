import sys

with open("metrics_3strike_v4.txt", "r", encoding="utf-16le", errors="ignore") as f:
    lines = f.readlines()
    
# Find the start of the metrics block
start_idx = -1
for i, line in enumerate(lines):
    if "--- 90-DAY METRICS RESULTS ---" in line:
        start_idx = i

if start_idx != -1:
    print("\n".join([l.strip() for l in lines[start_idx:]]))
else:
    print("Metrics block not found :(")
