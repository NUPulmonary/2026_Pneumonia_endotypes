import pickle

import pandas as pd

import common_data


def read_gene_dict():
    with open(f'{common_data.GENEFORMER_ROOT}/geneformer/token_dictionary.pkl', 'rb') as td_pkl:
        TOKEN_2_INDEX = pickle.load(td_pkl)

    GENE_FILE = f'{common_data.DATA}/02b_integration/09_raw/09_raw_genes_ensembl_ids.csv'
    GENE_2_TOKEN = pd.read_csv(GENE_FILE, index_col=0).gene_ids.to_dict()
    INDEX_2_TOKEN = {idx: token for token, idx in TOKEN_2_INDEX.items()}
    TOKEN_2_GENE = {token: gene for gene, token in GENE_2_TOKEN.items()}
    INDEX_2_GENE = {
        idx: TOKEN_2_GENE[INDEX_2_TOKEN[idx]]
        for idx in INDEX_2_TOKEN
        if INDEX_2_TOKEN[idx] in TOKEN_2_GENE
    }
    return INDEX_2_GENE
