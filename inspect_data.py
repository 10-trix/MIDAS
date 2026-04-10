import pandas as pd
import os

# Count parquet files
base = r'..\opencv\asl-signs\train_landmark_files'
total = 0
for d in os.listdir(base):
    total += len(os.listdir(os.path.join(base, d)))
print('Total parquet files:', total)

# Read train CSV
train = pd.read_csv(r'..\opencv\asl-signs\train.csv')
print('Train CSV rows:', len(train))
print('Unique signs:', train['sign'].nunique())
print('Signs sample:', sorted(train['sign'].unique())[:30])

# Check trainingData CSV format more
df_a = pd.read_csv(r'..\opencv\trainingData\A.csv')
print('\n--- trainingData/A.csv ---')
print('Shape:', df_a.shape)
print('Columns:', df_a.columns.tolist()[:5])
import ast
val = ast.literal_eval(df_a.iloc[0, 0])
print('Parsed value:', val, type(val))
print('42 cols = 21 landmarks x 2 hands => each cell is [x, y] pixel coords')

# Check how many files are actually available as parquet
available = set()
for d in os.listdir(base):
    for f in os.listdir(os.path.join(base, d)):
        seq_id = f.replace('.parquet', '')
        available.add(int(seq_id))
print(f'\nAvailable parquet files: {len(available)}')

# Match with train.csv
matched = train[train['sequence_id'].isin(available)]
print(f'Matched entries: {len(matched)}')
print(f'Signs in matched:', sorted(matched['sign'].unique())[:20])
