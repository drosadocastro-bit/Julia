import sqlite3
import pandas as pd
import json

conn = sqlite3.connect('test_metrics_temp.db')

# Print Sample of Actions
df = pd.read_sql("SELECT actions, reasoning FROM bitacora", conn)
print("--- ALL ACTIONS LOGGED ---")
types = set()
for i, row in df.iterrows():
    actions = json.loads(row['actions'])
    for a in actions:
        types.add(a.get("type"))

print(f"Unique Action Types Outputted over 90 Days: {types}")

    
# Print Mistakes
df_mistakes = pd.read_sql("SELECT * FROM mistakes", conn)
print(f"--- MISTAKES COUNT: {len(df_mistakes)} ---")

# Check if anything was flagged
df_flags = pd.read_sql("SELECT actions, reasoning FROM bitacora WHERE reasoning LIKE '%Invariant%' LIMIT 3", conn)
print(f"--- FIREWALL FLAGS: {len(df_flags)} ---")
