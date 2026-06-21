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


# python scripts/save_avg_mc_logits.py <tokens> <task> <split> <model> <mc_preds> <output.npy>

TOKENS = sys.argv[1]
TASK = sys.argv[2]
SPLIT = sys.argv[3]
MODEL = sys.argv[4]
MC_PATHS = sys.argv[5].split(' /// ')
LOGITS_PATH = sys.argv[6]


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


def load_logits(args):
    mc_dir, j = args
    file_path = f'{mc_dir}/logits_chunk{j}.npy'
    return np.load(file_path)


def stack_matrices(sublist):
    return scipy.sparse.vstack(sublist)


def get_mc_average_logits(model_path, dataset, gene_dict, mc_paths):
    matrices_list = []
    num_batches = 0
    while True:
        if os.path.exists(f'{mc_paths[0]}/logits_chunk{num_batches}.npy'):
            num_batches += 1
        else:
            break

    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [
            (mc_path, j)
            for mc_path in mc_paths for j in range(num_batches)
        ]
        results = list(tqdm.tqdm(executor.map(load_logits, tasks), total=len(tasks)))

        # Reshape the results list into a list of lists
        logits_list = [results[i:i+num_batches] for i in range(0, len(results), num_batches)]
        logits_list = [np.vstack(chunk_logits) for chunk_logits in logits_list]

        # get predictions
        predictions = np.mean(logits_list, axis=0)
        return predictions


def avg_and_save_mc_logits(
        task_info: common_data.TaskInfo, 
        gene_dict, 
        dataset,
        model_path,
        mc_paths,
        save_path
):
    avg_logits = get_mc_average_logits(
        model_path,
        dataset,
        gene_dict,
        mc_paths
    )
    np.save(save_path, avg_logits)


gene_dict = attention_utils.read_gene_dict()
data = load_data(TOKENS)
task_info = common_data.get_task_info(TASK, SPLIT)
test_data = filter_split_data(data, task_info)
avg_and_save_mc_logits(
    task_info, 
    gene_dict, 
    test_data, 
    MODEL,
    MC_PATHS,
    LOGITS_PATH
)

