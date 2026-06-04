# This cell loads the uploaded CSV, inspects it, and performs initial cleaning steps (strip columns, standardize booleans, parse dates)
import pandas as pd
import numpy as np
from tqdm import tqdm

# Read the CSV with provided encoding and parse dates where sensible
file_path = 'Untitled form (Responses) - Form responses 1.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Basic preview
print(df.head())

# Strip whitespace from column names
df.columns = [c.strip() for c in df.columns]

# Rename columns to simpler snake_case
rename_map = {
    'Timestamp': 'timestamp',
    'Your Name': 'name',
    'Which street or area in Andheri West are you giving feedback about?': 'area',
    'what do you feel about this street/area?(safe or unsafe)': 'safety_overall',
    'at what time of the day do you feel this area is safe?': 'safe_times',
    'at what time of the day do you feel this area is unsafe?': 'unsafe_times',
    'Is police patrolling carried out there?': 'police_patrolling',
    'is there CCTV surveillance in this area': 'cctv',
    'is the area free from catcalling, staring, or other forms of harassment?': 'harassment_free',
    'Have you ever witnessed or experienced any unsafe incident on this street?': 'unsafe_incident',
    'Suggestions to improve safety': 'suggestions'
}

# Handle potential extra spaces/hidden chars by mapping stripped keys
stripped_cols = {c.replace('\u00a0',' ').replace('\	',' ').replace('  ',' ').strip(): c for c in df.columns}
clean_map = {}
for k,v in rename_map.items():
    key = k.strip()
    if key in stripped_cols:
        clean_map[stripped_cols[key]] = v

df = df.rename(columns=clean_map)

# Parse timestamp
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True, infer_datetime_format=True)

# Trim strings across object columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str).str.strip()

# Normalize yes/no like fields to booleans where appropriate
bool_map = {
    'yes': True,
    'no': False,
    'not yet': False,
    'no not yet': False,
    'never noticed': np.nan,
    'never': np.nan,
    'not experienced': np.nan
}
for col in ['police_patrolling','cctv','harassment_free']:
    if col in df.columns:
        df[col + '_raw'] = df[col]
        df[col] = df[col].str.lower().map(bool_map)

# Create tidy categorical lists for safe/unsafe times
def split_times(s):
    if pd.isna(s):
        return []
    parts = [p.strip() for p in str(s).replace(';', ',').split(',') if len(p.strip())>0]
    return parts

for col in ['safe_times','unsafe_times']:
    if col in df.columns:
        df[col + '_list'] = df[col].apply(split_times)

# Standardize area text
if 'area' in df.columns:
    df['area_clean'] = df['area'].str.lower().str.replace('\s+', ' ', regex=True)

# Standardize safety_overall to categories
if 'safety_overall' in df.columns:
    df['safety_overall_clean'] = df['safety_overall'].str.lower()

# Show head of cleaned df
print(df.head())
print('Loaded and cleaned data head and columns shown above')