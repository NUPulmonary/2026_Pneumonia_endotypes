import sys
sys.path.insert(0, '../lib')

import os

import pandas as pd
import scanpy as sc

import common_data


# 'python scripts/write_loom.py "{input.data}" "{input.labels}" "{output}"'
DATA = sys.argv[1]
LABELS = sys.argv[2]
OUTPUT = sys.argv[3]


def copy_tasks(adata, labels):
    columns = common_data.get_task_columns(labels)
    na_values = common_data.na_values
    for column in columns:
        if pd.api.types.is_numeric_dtype(labels[column]):
            na_val = float(na_values.get(column, -1))
            vals = labels[column].fillna(na_val).astype(int)
        else:
            na_val = na_values.get(column, 'discard')
            vals = labels[column].fillna(na_val).astype(str)
            if column in adata.obs.columns:
                adata.obs[column] = adata.obs[column].astype(str)
        idx = adata.obs.individual.isin(vals.index)
        adata.obs.loc[idx, column] = vals[adata.obs.individual[idx]].values


def generate_looms(adata, labels, dataset_path):
    CHUNK_SIZE = 100_000
    adata_nrows = adata.shape[0]
    for chunk in range(adata_nrows // CHUNK_SIZE + 1):
        start = chunk * CHUNK_SIZE
        end = (chunk + 1) * CHUNK_SIZE if chunk < (adata_nrows // CHUNK_SIZE) else adata_nrows
        cells = adata.obs_names[start:end] # define the start and end indices for chunk

        fname = f'chunk_{chunk}'
        fdir = os.path.join(dataset_path, fname)  # Combine dataset_path and fname
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        adata_chunk = adata[adata.obs_names.isin(cells)].copy()
        copy_tasks(adata_chunk, labels)
        adata_chunk.write_loom(f'{fdir}/{fname}.loom')


labels = pd.read_csv(LABELS, index_col=0).set_index('bal_barcode')
adata = sc.read_h5ad(DATA)
adata = adata[:, ~adata.var.index.str.startswith("SARS-CoV-2")]
adata = adata[:, ~adata.var.index.str.startswith("MT-")]
# name columns according to Geneformer requirements
adata.var = adata.var.rename(columns={'total_counts': 'n_counts'})
adata.var = adata.var.rename(columns={'gene_ids': 'ensembl_id'})

generate_looms(adata, labels, OUTPUT)
