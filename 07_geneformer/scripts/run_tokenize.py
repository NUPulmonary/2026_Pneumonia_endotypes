import sys
sys.path.insert(0, '../lib')

import os

import geneformer
import pandas as pd

import common_data


# 'python scripts/tokenize.py "{input.loom}" "{input.labels}" "{output}"'
LOOM = sys.argv[1]
LABELS = sys.argv[2]
OUTPUT = sys.argv[3]

ADDITIONAL_COLUMNS = ['obs_names', 'patient', 'library_id', 'individual', 'Level_6']


def tokenize(tokenizer, loom_path, token_path, name):
    out = f'{token_path}'
    if not os.path.exists(out):
        os.makedirs(out)
    tokenizer.tokenize_data(loom_path, token_path, name)


labels = pd.read_csv(LABELS, index_col=0).set_index('bal_barcode')
tk = geneformer.TranscriptomeTokenizer(
    {c: c for c in ADDITIONAL_COLUMNS + common_data.get_task_columns(labels)},
    nproc=8
)

# Get a list of subdirectories (chunk folders) in loom_path
chunk_folders = [f for f in os.listdir(LOOM) if os.path.isdir(os.path.join(LOOM, f))]

# Loop through the chunk folders and tokenize them
for chunk in chunk_folders:
    loom_file_path = os.path.join(LOOM, chunk)
    tokenize(tk, loom_file_path, OUTPUT, chunk)
