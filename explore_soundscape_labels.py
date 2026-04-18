import pandas as pd

sl = pd.read_csv('data/birdclef-2026/birdclef-2026/train_soundscapes_labels.csv')
print('=== SHAPE ===')
print(f'Rows: {len(sl)}, Columns: {len(sl.columns)}')
print()
print('=== COLUMNS ===')
print(sl.columns.tolist())
print()
print('=== DTYPES ===')
print(sl.dtypes)
print()
print('=== FIRST 15 ROWS ===')
print(sl.head(15).to_string())
print()
print('=== UNIQUE VALUES ===')
for col in sl.columns:
    nuniq = sl[col].nunique()
    print(f'  {col}: {nuniq} unique')
print()

# Check time boundaries
time_cols = [c for c in sl.columns if 'time' in c.lower() or 'start' in c.lower() or 'end' in c.lower()]
if time_cols:
    print('=== TIME COLUMNS ===')
    for tc in time_cols:
        vals = sorted(sl[tc].unique()[:20])
        print(f'  {tc}: first 20 unique = {vals}')
    # Check durations if start/end pairs exist
    start_col = [c for c in time_cols if 'start' in c.lower()]
    end_col = [c for c in time_cols if 'end' in c.lower()]
    if start_col and end_col:
        durations = sl[end_col[0]] - sl[start_col[0]]
        print(f'  Duration distribution:\n{durations.value_counts().head(10)}')
print()

# Check species/label columns
label_cols = [c for c in sl.columns if 'label' in c.lower() or 'species' in c.lower()]
print('=== SPECIES INFO ===')
print(f'Label columns: {label_cols}')
for lc in label_cols:
    print(f'  {lc}: {sl[lc].nunique()} unique')
    print(f'  Sample values: {sl[lc].unique()[:15].tolist()}')
print()

# Check soundscape file references
file_cols = [c for c in sl.columns if 'file' in c.lower() or 'path' in c.lower() or 'name' in c.lower() or 'soundscape' in c.lower()]
if file_cols:
    print('=== FILE REFERENCES ===')
    for fc in file_cols:
        print(f'  {fc}: {sl[fc].nunique()} unique files')
        print(f'  Sample: {sl[fc].unique()[:5].tolist()}')

# Also check train.csv columns for comparison
print()
print('=== TRAIN.CSV FOR COMPARISON ===')
tr = pd.read_csv('data/birdclef-2026/birdclef-2026/train.csv')
print(f'Rows: {len(tr)}, Columns: {tr.columns.tolist()}')
print(f'Species: {tr["primary_label"].nunique()}')
print(tr.head(3).to_string())
