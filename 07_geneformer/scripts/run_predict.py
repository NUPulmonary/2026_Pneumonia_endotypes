import sys
sys.path.insert(0, '../lib')

import glob
import os
import pickle

import scipy.sparse
import datasets
import geneformer
import numpy as np
import pandas as pd
import transformers
import wandb
import torch
import inspect

import common_data
import geneformer_utils
import attention_utils


# 'python scripts/run_predict.py "{input.model}" "{input.tokens}" "{wildcards.task}" "{wildcards.split}" "{output}"'
MODEL = sys.argv[1]
TOKENS = sys.argv[2]
TASK = sys.argv[3]
SPLIT = sys.argv[4]
# output is the first MC path, so level up
OUTPUT = os.path.dirname(sys.argv[5])

IS_TRAIN = False
if len(sys.argv) > 6 and sys.argv[6] == 'train':
    IS_TRAIN = True

NUM_MC_SAMPLES = 16


def load_data(token_path):
    # Filter files with prefix "chunk"
    chunks = glob.glob(f'{token_path}/chunk*.dataset')
    tokenized_chunks = [datasets.load_from_disk(file) for file in chunks]
    return datasets.concatenate_datasets(tokenized_chunks)


def filter_split_data(data, task: common_data.TaskInfo, is_train: bool):
    data_label = 'train' if is_train else 'test'
    task_column = task
    idx = np.isin(data[task.column], task.column_values)
    test_idx = idx & np.char.equal(data[task.split_column], data_label)
    data = data.rename_column(task.column, 'label')

    test = data.select(np.where(test_idx)[0])

    if task.column_values != [0, 1]:
        values_to_labels = {v: i for i, v in enumerate(task.column_values)}
        test = test.map(lambda x: {'label': values_to_labels[x['label']]}, num_proc=16)

    return test


def save_and_reset_mc_chunk(accumulated_result, chunk_num, path, mc_samples):
    for i in range(mc_samples):
        mc_dir = f'{path}/mc_{i}'
        os.makedirs(mc_dir, exist_ok=True)
        mc_attn = scipy.sparse.vstack(accumulated_result[i][0])
        scipy.sparse.save_npz(f'{mc_dir}/attn_chunk{chunk_num}.npz', mc_attn)
        mc_logits = np.vstack(accumulated_result[i][2])
        np.save(f'{mc_dir}/logits_chunk{chunk_num}.npy', mc_logits)
        if i == 0:
            mc_rank = scipy.sparse.vstack(accumulated_result[i][1])
            scipy.sparse.save_npz(f'{mc_dir}/rank_chunk{chunk_num}.npz', mc_rank)
        # reset corresponding MC accumulation memory
        accumulated_result[i] = [[], [], []]


def forward_and_save_attn(
        model,
        dataloader,
        n_total_genes,
        save_path,
        multiply_by_n_tokens=False,
        mc_samples=16,
        chunk_size=10_000
):
    chunk_num = 0
    full_result = []
    for i in range(mc_samples):
        # 0: attn, 1: ranks, 2: logits
        full_result.append([[], [], []])
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            gene_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            gene_ids, attention_mask = gene_ids.to('cuda'), attention_mask.to('cuda')
            batch['input_ids'] = gene_ids
            batch['attention_mask'] = attention_mask
            for i in range(mc_samples):
                attn, rank, logits = geneformer_utils.forward_and_return_attn(
                    model,
                    batch,
                    n_total_genes,
                    multiply_by_n_tokens
                )
                full_result[i][0].extend(attn)
                full_result[i][2].extend(logits)
                if i == 0:
                    full_result[i][1].extend(rank)
            # save chunk if larger than `chunk_size`
            if len(full_result[0][0]) > chunk_size:
                save_and_reset_mc_chunk(full_result, chunk_num, save_path, mc_samples)
                chunk_num += 1
    # save the rest
    save_and_reset_mc_chunk(full_result, chunk_num, save_path, mc_samples)


rng = np.random.default_rng(1066)
task = common_data.get_task_info(TASK, SPLIT)
data = load_data(TOKENS)
test = filter_split_data(data, task, IS_TRAIN)

test = test.sort('length')
# Save cells because we reordered them
tmp_output = OUTPUT + '.tmp'
os.makedirs(tmp_output, exist_ok=True)
np.save(f'{tmp_output}/cell_ids', test['obs_names'])

print(f'Test size: {len(test)}')

model = geneformer_utils.get_model(path=MODEL, output_attentions=True, enable_random_dropout=True)

# Mimick huggingface.Trainer `get_eval_dataloader`
signature_columns = set(inspect.signature(model.forward).parameters.keys())
signature_columns |= {'label', 'label_ids'}
unused_columns = list(
    set(test.column_names)
    - signature_columns
)
unused_columns = [col for col in unused_columns if col in test.column_names]
test = test.remove_columns(unused_columns)
data_loader = torch.utils.data.dataloader.DataLoader(
    test,
    batch_size=64,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    collate_fn=geneformer.DataCollatorForCellClassification()
)

n_total_genes = max(attention_utils.read_gene_dict().keys()) + 1

forward_and_save_attn(
    model,
    data_loader,
    n_total_genes,
    save_path=tmp_output,
    multiply_by_n_tokens=True,
    mc_samples=NUM_MC_SAMPLES
)

os.rename(tmp_output, OUTPUT)
