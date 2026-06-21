import sys
sys.path.insert(0, '../lib')

import concurrent.futures
import gc
import glob
import math
import os
import pickle

import scipy.sparse
import datasets
import geneformer
import numpy as np
import pandas as pd
import tqdm

import common_data
import geneformer_utils
import attention_utils


# python scripts/save_avg_mc_attn.py <tokens> <task> <split> <model> <mc_preds> <output.npz> <output.csv>

TOKENS = sys.argv[1]
TASK = sys.argv[2]
SPLIT = sys.argv[3]
MODEL = sys.argv[4]
MC_PATHS = sys.argv[5].split(' /// ')
ATTN_MTX_PATH = sys.argv[6]
ATTN_GENES_PATH = sys.argv[7]


def load_data(token_path):
    # Filter files with prefix "chunk"
    chunks = glob.glob(f'{token_path}/chunk*.dataset')
    tokenized_chunks = [datasets.load_from_disk(file) for file in chunks]
    return datasets.concatenate_datasets(tokenized_chunks)


def filter_split_data(data, task: common_data.TaskInfo):
    task_column = task
    idx = np.isin(data[task.column], task.column_values)
    test_idx = idx & np.char.equal(data[task.split_column], 'test')
    data = data.rename_column(task.column, 'label')

    test = data.select(np.where(test_idx)[0])

    if task.column_values != [0, 1]:
        values_to_labels = {v: i for i, v in enumerate(task.column_values)}
        test = test.map(lambda x: {'label': values_to_labels[x['label']]}, num_proc=16)

    return test


def load_matrix(args):
    mc_dir, j = args
    file_path = f'{mc_dir}/attn_chunk{j}.npz'
    return scipy.sparse.load_npz(file_path).tocsr()


def stack_matrices(sublist):
    return scipy.sparse.vstack(sublist)


def get_mc_average_attn(model_path, dataset, gene_dict, mc_paths):
    matrices_list = []
    num_batches = 0
    while True:
        if os.path.exists(f'{mc_paths[0]}/attn_chunk{num_batches}.npz'):
            num_batches += 1
        else:
            break

    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [
            (mc_path, j)
            for mc_path in mc_paths for j in range(num_batches)
        ]
        results = list(tqdm.tqdm(executor.map(load_matrix, tasks), total=len(tasks)))

    # Reshape the results list into a list of lists
    matrices_list = [results[i:i + num_batches] for i in range(0, len(results), num_batches)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        matrices_list = list(tqdm.tqdm(
            executor.map(stack_matrices, matrices_list),
            total=len(matrices_list)
        ))
    del results
    gc.collect()

    sum_matrix = matrices_list[0]

    # Sum up the matrices element-wise
    for matrix in tqdm.tqdm(matrices_list[1:], desc="Summing matrices"):
        sum_matrix += matrix

    # Divide each element in the sum_matrix by the number of matrices to get the mean
    average_matrix = sum_matrix / len(matrices_list)
    out_attn = average_matrix.tocsc()

    del matrices_list
    del average_matrix
    gc.collect()

    n_cells_per_gene = (out_attn > 0).sum(axis=0).A1
    gene_names = np.array([gene_dict.get(i, 'dummy') for i in range(max(gene_dict.keys()) + 1)])
    out_attn = out_attn[:, n_cells_per_gene > 0]
    gene_names = gene_names[n_cells_per_gene > 0]
    return out_attn, gene_names


def correct_attn_weights(out_attn, cell_lengths):
    actual_cell_sum = out_attn.sum(axis=1).A1
    expected_cell_sum = np.array(cell_lengths)
    ratio = np.reshape((expected_cell_sum / actual_cell_sum), (out_attn.shape[0], 1))
    return out_attn.multiply(ratio).tocsr()


def avg_and_save_mc_attention(
        task_info: common_data.TaskInfo, 
        gene_dict, 
        dataset,
        model_path,
        mc_paths,
        save_path,
        save_genes_path
):
    out_attn, gene_names = get_mc_average_attn(
        model_path,
        dataset,
        gene_dict,
        mc_paths
    )
    out_attn = correct_attn_weights(out_attn, dataset['length'])
    scipy.sparse.save_npz(save_path, out_attn)
    pd.Series(gene_names).to_csv(save_genes_path)


gene_dict = attention_utils.read_gene_dict()
data = load_data(TOKENS)
task_info = common_data.get_task_info(TASK, SPLIT)
test_data = filter_split_data(data, task_info)
avg_and_save_mc_attention(
    task_info, 
    gene_dict, 
    test_data, 
    MODEL,
    MC_PATHS,
    ATTN_MTX_PATH,
    ATTN_GENES_PATH
)

