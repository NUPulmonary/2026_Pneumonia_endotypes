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
TOKENS = sys.argv[1]
# output is the first MC path, so level up
OUTPUT = os.path.dirname(sys.argv[2])



def load_data(token_path):
    # Filter files with prefix "chunk"
    chunks = glob.glob(f'{token_path}/chunk*.dataset')
    tokenized_chunks = [datasets.load_from_disk(file) for file in chunks]
    return datasets.concatenate_datasets(tokenized_chunks)

# Isolate the sample ids of the rare pathogen samples
sc_labels = pd.read_csv(common_data.SC_LABELS)
our_samples = list(sc_labels.bal_barcode[sc_labels.cohort.isin(['LongCOVID'])])

def filter_split_data(data):
    idx = []
    for individual in data['individual']:
        idx.append(individual in our_samples)
    test = data.select(np.where(np.array(idx))[0])
    test = test.add_column('label', [1] * len(test))
    return test


def save_and_reset_mc_chunk(accumulated_result, chunk_num, path, mc_samples):
    os.makedirs(path, exist_ok=True)
    mc_logits = np.vstack(accumulated_result[0])
    np.save(f'{path}/long_covid_cell_embeddings.npy', mc_logits)
    # reset corresponding MC accumulation memory
    accumulated_result[0] = []


def forward_and_save(
        model,
        dataloader,
        n_total_genes,
        save_path,
        multiply_by_n_tokens=False,
        mc_samples=16,
        chunk_size=10_000
):
    chunk_num = 0
    full_result = [[]]
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            gene_ids, attention_mask = batch['input_ids'], batch['attention_mask']
            gene_ids, attention_mask = gene_ids.to('cuda'), attention_mask.to('cuda')
            model_predictions = model(
                input_ids=gene_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=True
            )
            embeddings = list(model_predictions['hidden_states'][-2].mean(axis=1).cpu().detach().numpy())
            full_result[0].extend(embeddings)

            # save chunk if larger than `chunk_size`
            # if len(full_result[0]) > chunk_size:
            #     save_and_reset_mc_chunk(full_result, chunk_num, save_path, mc_samples)
            #     chunk_num += 1
            #     print('CHUNK', flush=True)
    # save the rest
    save_and_reset_mc_chunk(full_result, chunk_num, save_path, mc_samples)


rng = np.random.default_rng(1066)
data = load_data(TOKENS)
test = filter_split_data(data)

test = test.sort('length')
# Save cells because we reordered them
tmp_output = OUTPUT
os.makedirs(tmp_output, exist_ok=True)
np.save(f'{tmp_output}/long_covid_cell_ids.npy', test['obs_names'])

print(f'Test size: {len(test)}', flush=True)

model = geneformer_utils.get_model(output_attentions=False, enable_random_dropout=False, output_hidden_states=True)

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

forward_and_save(
    model,
    data_loader,
    n_total_genes,
    save_path=tmp_output,
    multiply_by_n_tokens=True,
    mc_samples=0
)

#os.rename(tmp_output, OUTPUT)

