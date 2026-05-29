import sys
sys.path.insert(0, '../lib')

import os
import pickle
import numpy as np
import torch
import pandas as pd
import common_data

from datasets import Dataset
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, Trainer


#torch.manual_seed(42)
#torch.cuda.manual_seed_all(42)

# TORCH MISC
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512

# DATA MISC
NB_OF_BINS = 4 + 1 # Number of bins into which data values were binned (4) + 1 for NA
IS_NA_BIN_NAME = 'binis_na' # Name of the column for NA bin, this changed because of quartile binning
IS_NA_BIN_TOKEN = NB_OF_BINS + 2 - 1

# PATH MISC
PATOGENE_LABELS_PATH = common_data.CLINICAL_LABELS
EHRFORMER_SAVING_PATH = '../data/01_ehrformer/models/pretrained/'

# TOKEN_MISC
CLS_TOKEN = 0
MASK_TOKEN = 1


# MODELS MISC
FREEZE_FIRST_ENCODER_LAYERS = 1
SAVE_BASE_MODEL = True
TRAIN_BASE = True
MASK_PROB = 0.15
# TRAIN_FINETUNNED = False

RAND = torch.Generator().manual_seed(9102730192)



EHR_FILE_PATH = common_data.EHRFORMER_BINS

with open(EHR_FILE_PATH, 'rb') as pickle_handle:
    ehr_df = pickle.load(pickle_handle)

NB_OF_DATA_COLUMNS = len(ehr_df.columns) // 5

labels_df = pd.read_csv(PATOGENE_LABELS_PATH)

splits_columns = [
    col
    for col in labels_df.columns
    if np.any([col.startswith(prefix) for prefix in common_data.TASKS.values()])
]

# define all 12 train-test splits for the pretrained model, there is probably a better way to do this

train_splits_list = []
test_splits_list = []

for col in splits_columns:
    TEST_PATIENTS_IDS = labels_df[labels_df[col] == 'test'] # split 1
    NA_PATIENTS_IDS = labels_df[labels_df[col].isna()] # split 1
    NA_PATIENTS_IDS = NA_PATIENTS_IDS[~NA_PATIENTS_IDS['patient'].isin(TEST_PATIENTS_IDS['patient'])]
    TRAIN_PATIENTS_IDS = labels_df[labels_df[col] == 'train']
    dfs_to_concat = [TRAIN_PATIENTS_IDS, NA_PATIENTS_IDS]
    TRAIN_PATIENTS_IDS = pd.concat(dfs_to_concat)
    TRAIN_PATIENTS_IDS = list(set(TRAIN_PATIENTS_IDS['patient']))
    TEST_PATIENTS_IDS = list(set(TEST_PATIENTS_IDS['patient']))
    train_splits_list.append(TRAIN_PATIENTS_IDS)
    test_splits_list.append(TEST_PATIENTS_IDS)


patients = np.array([
    index_.split('/')[0]
    for index_ in ehr_df.index
])

patient_test_set_list = []
patient_train_set_list = []

for l in train_splits_list:
    patient_train_set = [
        str(patient_int)
        for patient_int in l
    ]
    patient_train_set_list.append(patient_train_set)

for l in test_splits_list:
    patient_test_set = [
        str(patient_int)
        for patient_int in l
    ]
    patient_test_set_list.append(patient_test_set)

train_indices_list = []

for l in patient_train_set_list:
    train_indices = np.array([
        index_ for index_, patient in enumerate(patients)
        if patient in l
    ])
    train_indices_list.append(train_indices)

test_indices_list = []

for l in patient_test_set_list:
    test_indices = np.array([
        index_ for index_, patient in enumerate(patients)
        if patient in l
    ])
    test_indices_list.append(test_indices)


ehr_bins = [
    [
        column_name[1]
        for column_name, column_value in zip(ehr_df.columns, row_)
        if column_value
    ]
    for row_nb, row_ in ehr_df.iterrows()
]

ehr_bins_train_list = []
for train_list in train_indices_list:
    ehr_bins_train = [ehr_bins[train_index] for train_index in train_list]
    ehr_bins_train_list.append(ehr_bins_train)

ehr_bins_test_list = []
for test_list in test_indices_list:
    ehr_bins_test = [ehr_bins[test_index] for test_index in test_list]
    ehr_bins_test_list.append(ehr_bins_test)


VOCAB = {
    "CLS": CLS_TOKEN,
    "MASK": MASK_TOKEN,
    **{
        f'bin{ind_}': ind_ + 2 for ind_ in range(NB_OF_BINS - 1)
    },
    IS_NA_BIN_NAME: IS_NA_BIN_TOKEN
}

ehr_row_tokens_train_list = []
for l in ehr_bins_train_list:
    ehr_row_tokens_train = [
    [VOCAB[token] for token in ehr_row]
    for ehr_row in l
    ]
    ehr_row_tokens_train_list.append(ehr_row_tokens_train)

ehr_row_tokens_test_list = []
for l in ehr_bins_test_list:
    ehr_row_tokens_test = [
    [VOCAB[token] for token in ehr_row]
    for ehr_row in l
    ]
    ehr_row_tokens_test_list.append(ehr_row_tokens_test)

ehr_row_dataset_train_list = []

for l in ehr_row_tokens_train_list:
    l = Dataset.from_dict({'input_ids': l})
    ehr_row_dataset_train_list.append(l)

ehr_row_dataset_test_list = []

for l in ehr_row_tokens_test_list:
    l = Dataset.from_dict({'input_ids': l})
    ehr_row_dataset_test_list.append(l)


class EHRMaskedLMFormerDataCollator:
    def __init__(
        self,
        cls_token=CLS_TOKEN,
        mask_token=MASK_TOKEN,
        mask_prob=MASK_PROB,
    ):
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.mask_prob = mask_prob

    def __call__(self, ehrs):
        tokens = torch.tensor([
            [self.cls_token] + ehr_row['input_ids']
            for ehr_row in ehrs
        ])
        masks = []
        for ehr_row in ehrs:
            input_ids = ehr_row['input_ids']
            # Initialize a mask with zeros, +1 for class token
            mask = torch.zeros(len(input_ids) + 1, dtype=torch.bool)
            # shift the start of the masking to the second index to account for adding the class token
            for i in range(1, len(input_ids) + 1):
                token_id = input_ids[i - 1]
                # do not mask class token, do not mask NA
                if token_id == CLS_TOKEN or token_id == IS_NA_BIN_TOKEN:
                    continue
                if torch.rand(1, generator=RAND) < self.mask_prob:
                    mask[i] = 1
                else:
                    mask[i] = 0
            masks.append(mask)

        masks = torch.stack(masks)

        labels = tokens.clone()

        tokens[masks] = self.mask_token
        # Replace unmasked indices with -100 in the labels since we only compute loss on masked tokens
        # from https://github.com/huggingface/transformers/blob/93aafdc620d39b9ec714ffecf015a085ea221282/src/transformers/data/data_collator.py#L749C9-L749C103
        labels[torch.logical_not(masks)] = -100
        return {'input_ids': tokens, 'labels': labels}



# model type
model_type = "bert"
# max input size
max_input_size = 1 + NB_OF_DATA_COLUMNS  # + CLS Token
# number of layers
num_layers = 4
# number of attention heads
num_attn_heads = 4
# number of embedding dimensions
num_embed_dim = 64
# intermediate size
intermed_size = num_embed_dim * 2
# activation function
activ_fn = "relu"
# initializer range, layer norm, dropout
initializer_range = 0.02
layer_norm_eps = 1e-12
attention_probs_dropout_prob = 0.2
hidden_dropout_prob = 0.2


masked_lm_config_dict = {
    "hidden_size": num_embed_dim,
    "num_hidden_layers": num_layers,
    "initializer_range": initializer_range,
    "layer_norm_eps": layer_norm_eps,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "hidden_dropout_prob": hidden_dropout_prob,
    "intermediate_size": intermed_size,
    "hidden_act": activ_fn,
    "max_position_embeddings": max_input_size,
    "model_type": model_type,
    "num_attention_heads": num_attn_heads,
    "pad_token_id": None,
    "vocab_size": len(VOCAB)  # bins+2 for <mask> and <pad> tokens
}

masked_lm_config = BertConfig(**masked_lm_config_dict)


ehr_former = BertForMaskedLM(masked_lm_config)
ehr_former.train()

masked_lm_training_args_dict = {
    "learning_rate": 5e-5,
    "do_train": True,
    "do_eval": True,
    "save_strategy": "epoch",
    "report_to": "none",
    "logging_steps": 5,
    "evaluation_strategy": "epoch",
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": 'linear',
    "warmup_steps": 10,
    "weight_decay": 0.001,
    "per_device_train_batch_size": BATCH_SIZE,
    "num_train_epochs": 300,
    "output_dir": './model_output_1',
}

masked_lm_training_args = TrainingArguments(**masked_lm_training_args_dict)

for i in range(len(ehr_row_dataset_train_list)):
    if os.path.exists(EHRFORMER_SAVING_PATH + splits_columns[i]):
        print('-' * 50)
        print(f'[!] Model for {splits_columns[i]} exists, not retraining')
        print('-' * 50)
        print('')
        continue

    print('-' * 50)
    print(f'[!] Starting on {splits_columns[i]}')
    print('-' * 50)
    print('')

    data_collator = EHRMaskedLMFormerDataCollator()

    masked_lm_trainer = Trainer(
            model=ehr_former,
            args=masked_lm_training_args,
            train_dataset=ehr_row_dataset_train_list[i],
            eval_dataset=ehr_row_dataset_test_list[i],
            data_collator=data_collator,
        )

    if TRAIN_BASE:
        masked_lm_trainer.train()

    if SAVE_BASE_MODEL:
        masked_lm_trainer.save_model(EHRFORMER_SAVING_PATH + splits_columns[i])
        processed_data = data_collator(ehr_row_dataset_train_list[i]) # save the mask tensor???
        torch.save(processed_data, EHRFORMER_SAVING_PATH + 'mask_tensor/' + splits_columns[i] + '.pt')
