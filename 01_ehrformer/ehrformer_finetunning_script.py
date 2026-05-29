import sys
sys.path.insert(0, '../lib')

import argparse
import numpy as np
import os
import pandas
import torch
import warnings

from datasets import Dataset
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import transformers
import common_data
import geneformer_utils
import wandb
import scipy
import sklearn.metrics


N_MC_SAMPLES = 16

# TORCH MISC
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TAKE_EVERY = 1

# DATA MISC
NB_OF_BINS = 4 + 1 # Number of bins into which data values were binned (4) + 1 for NA
IS_NA_BIN_NAME = 'binis_na' # Name of the column for NA bin

# PATH MISC
EHR_FILE_PATH = common_data.EHRFORMER_BINS
PATOGENE_LABELS_PATH = common_data.CLINICAL_LABELS

PATHOGEN_LIST = [
    'Early SARS-CoV-2; Gram+',
    'Late SARS-CoV-2',
    'Late SARS-CoV-2; Gram+',
    'Early SARS-CoV-2',
    'Pseudomonas aeruginosa; SARS-CoV-2',
    'Gram-*; Gram+',
    'Gram-*',
    'Gram+',
    'Pseudomonas aeruginosa',
    'NPC',
]

PATHOGEN_TASK_NAME = 'perturbation_groups_2'

# TOKEN_MISC
CLS_TOKEN = 0
MASK_TOKEN = 1

# MODELS MISC
FREEZE_FIRST_ENCODER_LAYERS = 3


class EHRSequenceClassificationFormerDataCollator:
    def __init__(
        self,
        cls_token=CLS_TOKEN,
        mask_token=MASK_TOKEN,
    ):
        self.cls_token = cls_token
        self.mask_token = mask_token

    def __call__(self, ehrs):
        tokens = torch.tensor([
            [self.cls_token] + ehr_row['input_ids']
            for ehr_row in ehrs
        ])
        labels = torch.tensor([
            [ehr_row['labels']]
            for ehr_row in ehrs
        ]).to(torch.float)
        return {
            'input_ids': tokens,
            'labels': labels
        }


def compute_metrics(pred):
    logits, y_true = pred.predictions, pred.label_ids
    y_true = y_true[:, 0]
    y_probs = scipy.special.expit(logits)
    y_pred = y_probs[:, 0] > 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auprc = sklearn.metrics.average_precision_score(y_true, y_probs[:, 0])
        f1_binary = sklearn.metrics.f1_score(y_true, y_pred, average='binary')
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_probs[:, 0])
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred)
        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred)

    return {
        'accuracy': accuracy,
        'f1_binary': f1_binary,
        'AUROC': roc_auc,
        'AUPRC': auprc,
        'recall': recall,
        'precision': precision,
    }


def compute_model_attention_weights_for_ids(
    attentions,
    attention_of_indices=0,
):
    attentions_sumed_over_heads = [att_.mean(axis=1) for att_ in attentions]
    attention_weights = geneformer_utils.aggregate_attention_weights(
        attentions_sumed_over_heads,
        attention_of_indices
    )
    return attention_weights


def prepare_data_for_category(category, ehr_data_full, pl, ehr_columns, VOCAB, split_name, task_name):
    assert ehr_data_full.shape[0] == pl.shape[0]
    train_xs, train_ys, test_xs, test_ys, train_patient_id, test_patient_id = [], [], [], [], [], []

    for (_, ehr_row), (_, pl_row) in zip(ehr_data_full.iterrows(), pl.iterrows()):
        current_train_test_flag = pl_row[split_name]
        current_target = pl_row[task_name]

        if (
            current_target == -1
            or (
                task_name == PATHOGEN_TASK_NAME
                and current_target != category
                and category != 'pathogen_all'
                and current_target != 'NPC'
            )
        ):
            continue

        if task_name == PATHOGEN_TASK_NAME:
            if category == 'pathogen_all':
                binary_target = 0 if current_target == 'NPC' else 1
            else:
                binary_target = 1 if current_target == category else 0
        else:
            binary_target = current_target

        current_bins = [
            column_name[1]
            for column_name, column_value
            in zip(ehr_columns, ehr_row)
            if column_value
        ]

        if current_train_test_flag == 'train':
            train_xs.append(current_bins)
            train_ys.append(binary_target)
            train_patient_id.append(pl_row['Unnamed: 0'])
        elif current_train_test_flag == 'test':
            test_xs.append(current_bins)
            test_ys.append(binary_target)
            test_patient_id.append(pl_row['Unnamed: 0'])

    ehr_row_tokens_pathogen_train = [[VOCAB[token] for token in ehr_row] for ehr_row in train_xs]
    ehr_row_tokens_pathogen_test = [[VOCAB[token] for token in ehr_row] for ehr_row in test_xs]
    # named tuple
    return (
        ehr_row_tokens_pathogen_train,
        ehr_row_tokens_pathogen_test,
        train_ys,
        test_ys,
        train_patient_id,
        test_patient_id
    )


def init_wandb(task_name, selected_category, split_name, OUTPUT, params, model, train_data, eval_data):
    hyperparameters = params.copy()
    hyperparameters['freeze_layers'] = FREEZE_FIRST_ENCODER_LAYERS
    hyperparameters['freeze_embeddings'] = True
    hyperparameters['architecture'] = 'EHRformer_4l'
    hyperparameters['train/n_days_class_0'] = np.equal(train_data, 0).sum()
    hyperparameters['train/n_days_class_1'] = np.equal(train_data, 1).sum()
    hyperparameters['eval/n_days_class_0'] = np.equal(eval_data, 0).sum()
    hyperparameters['eval/n_days_class_1'] = np.equal(eval_data, 1).sum()
    hyperparameters.update(model.config.to_diff_dict())

    group_name = 'EHRrc2_' + task_name
    if selected_category != task_name:
        group_name += f'_{selected_category}'
    model_name = f'{group_name}_{split_name}_{os.path.basename(OUTPUT).replace("model_", "")}'
    wandb.init(
        project="serniczek",
        group=group_name,
        name=model_name,
        config=hyperparameters,
    )


@torch.no_grad()
def predict_with_mc(model, batch, num_mc_samples):
    gene_ids = batch['input_ids']

    output_logits = torch.zeros(size=(gene_ids.shape[0], num_mc_samples)).to(DEVICE)
    output_attentions = torch.zeros(size=(gene_ids.shape[0], num_mc_samples, gene_ids.shape[1])).to(DEVICE)
    for mc_sample in range(num_mc_samples):
        model_predictions = model(
            input_ids=gene_ids,
            output_attentions=True,
            output_hidden_states=False,
        )
        output_logits[:, mc_sample] = model_predictions['logits'][:, 0]
        output_attentions[:, mc_sample, :] = compute_model_attention_weights_for_ids(
            model_predictions['attentions']
        )[:, 0]
    return output_logits, output_attentions


def main(
    selected_category: str,
    split_name: str,
    run_nb: int,
    pretrained_path: str,
    task_name: str,
    epoch_nb: int,
):
    OUTPUT = '../data/01_ehrformer/models/finetuned_rc2/'
    if selected_category == task_name:
        OUTPUT += f'{task_name}/{split_name}/model_{run_nb}/'
    else:
        OUTPUT += f'{task_name}_{selected_category}/{split_name}/model_{run_nb}'
    print(f'Saving to {OUTPUT}')
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    with open(EHR_FILE_PATH, 'rb') as pickle_handle:
        ehr_df = pickle.load(pickle_handle)

    pl = pandas.read_csv(PATOGENE_LABELS_PATH)

    new_pl = pl.copy()
    new_pl = new_pl.rename(columns={'Unnamed: 0': 'merge_key'})
    new_pl['merge_key'] = (
        new_pl.patient.astype(str)
        + '-'
        + new_pl.ICU_stay.astype(str)
        + '-'
        + new_pl.ICU_day.astype(str)
    )

    ehr_df_flatten = ehr_df.copy()
    ehr_df_flatten.columns = ['&'.join(col).strip() for col in ehr_df_flatten.columns.values]
    ehr_df_flatten = ehr_df_flatten.reset_index().rename(columns={'index': 'merge_key'})
    ehr_df_flatten['merge_key'] = ehr_df_flatten['merge_key'].str.replace('/', '-')

    full_df = pandas.merge(ehr_df_flatten, new_pl, on='merge_key', how='inner')

    # if task_name == PATHOGEN_TASK_NAME:
    # 	full_df = full_df[full_df[task_name].isin(PATHOGEN_LIST)]

    ehr_columns = []
    ehr_columns_full = []

    for col in full_df.columns:
        if '&' in col:
            current_value = tuple(col.split('&'))
            if current_value in ehr_df.columns:
            #ehr_columns.append(tuple(col.split('&')))
                ehr_columns.append(current_value)
                ehr_columns_full.append(col)


    ehr_data_full = full_df[ehr_columns_full]
    ehr_data_full.columns = ehr_columns

    VOCAB = {
        "CLS": CLS_TOKEN,
        "MASK": MASK_TOKEN,
        **{
            f'bin{ind_}': ind_ + 2 for ind_ in range(NB_OF_BINS - 1)
        },
        IS_NA_BIN_NAME: NB_OF_BINS + 2 - 1
    }

    finetunned_ehr_former = BertForSequenceClassification.from_pretrained(
        pretrained_path,
        problem_type="multi_label_classification",
        num_labels=1,
    ).to(DEVICE)
    new_classifier = torch.nn.Sequential(
        torch.nn.LayerNorm(finetunned_ehr_former.config.hidden_size),
        torch.nn.Linear(finetunned_ehr_former.config.hidden_size, finetunned_ehr_former.config.hidden_size * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(finetunned_ehr_former.config.hidden_size * 2, finetunned_ehr_former.config.hidden_size),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(finetunned_ehr_former.config.hidden_size),
        torch.nn.Linear(finetunned_ehr_former.config.hidden_size, finetunned_ehr_former.config.num_labels)
    )
    finetunned_ehr_former.classifier = new_classifier

    geneformer_utils.freeze_encoder_layers(finetunned_ehr_former, layers=FREEZE_FIRST_ENCODER_LAYERS)
    geneformer_utils.freeze_model_embeddings(finetunned_ehr_former)
    finetunned_ehr_former.train()

    training_args_dict = {
        "learning_rate": 1e-4,
        "do_train": True,
        "do_eval": True,
        "save_strategy": "epoch",
        "logging_steps": 2,
        "evaluation_strategy": "epoch",
        "disable_tqdm": False,
        "lr_scheduler_type": 'constant',
        "warmup_steps": 0,
        "weight_decay": 0.01, # default 0
        "max_grad_norm": 1, # default 1
        "per_device_train_batch_size": BATCH_SIZE,
        "num_train_epochs": epoch_nb,
        "output_dir": OUTPUT,
        "report_to": 'wandb'
    }
    # For VAP tasks, change weight_decay and learning_rate
    if task_name != PATHOGEN_TASK_NAME:
        training_args_dict['learning_rate'] = 4e-5
        training_args_dict['weight_decay'] = 0.05
    training_args = TrainingArguments(**training_args_dict)

    (
        ehr_row_tokens_pathogen_train,
        ehr_row_tokens_pathogen_test,
        train_ys,
        test_ys,
        train_patient_id,
        test_patient_id
    ) = prepare_data_for_category(
        selected_category,
        ehr_data_full,
        pl,
        ehr_columns,
        VOCAB,
        split_name,
        task_name
    )

    ehr_row_dataset_pat_train = Dataset.from_dict({
        'input_ids': ehr_row_tokens_pathogen_train[::TAKE_EVERY],
        'labels': train_ys[::TAKE_EVERY]
    })
    ehr_row_dataset_pat_test = Dataset.from_dict({
        'input_ids': ehr_row_tokens_pathogen_test[::TAKE_EVERY],
        'labels': test_ys[::TAKE_EVERY]
    })
    # print(f'selected category: {selected_category}')
    # print(f'ehr data full shape: {ehr_data_full.shape}')
    # print(f'labels shape: {pl.shape}')
    print(f'N train: {len(train_ys)}; N test: {len(test_ys)}')
    print(f'Train TRUE percentage: {np.mean(train_ys[::TAKE_EVERY])}, test: {np.mean(test_ys[::TAKE_EVERY])}')

    init_wandb(
        task_name,
        selected_category,
        split_name,
        OUTPUT,
        training_args_dict,
        finetunned_ehr_former,
        train_ys,
        test_ys
    )

    fine_tunning_trainer = Trainer(
        model=finetunned_ehr_former,
        args=training_args,
        train_dataset=ehr_row_dataset_pat_train,
        eval_dataset=ehr_row_dataset_pat_test,
        data_collator=EHRSequenceClassificationFormerDataCollator(),
        compute_metrics=compute_metrics,
    )

    for cb in fine_tunning_trainer.callback_handler.callbacks:
        if isinstance(cb, transformers.integrations.NeptuneCallback):
            fine_tunning_trainer.callback_handler.remove_callback(cb)

    fine_tunning_trainer.train()
    fine_tunning_trainer.save_model(OUTPUT)

    test_preds = fine_tunning_trainer.predict(ehr_row_dataset_pat_test)
    with open(f'{OUTPUT}/predictions_test.pkl', 'wb') as fp:
        pickle.dump(test_preds, fp)
    train_preds = fine_tunning_trainer.predict(ehr_row_dataset_pat_train)
    with open(f'{OUTPUT}/predictions_train.pkl', 'wb') as fp:
        pickle.dump(train_preds, fp)

    # Run MC predictions and save
    fine_tunning_trainer._get_train_sampler = lambda: None
    finetunned_ehr_former.train()

    train_data_loader = fine_tunning_trainer.get_train_dataloader()
    train_logits = []
    train_attn = []
    for batch in train_data_loader:
         l, a = predict_with_mc(
             finetunned_ehr_former,
             batch,
             N_MC_SAMPLES
         )
         train_logits.append(l)
         train_attn.append(a)
    train_logits = torch.concat(train_logits).cpu().detach().numpy()
    train_attn = torch.concat(train_attn).cpu().detach().numpy()
    np.save(f'{OUTPUT}/mc_train_logits.npy', train_logits)
    np.save(f'{OUTPUT}/mc_train_attn.npy', train_attn)

    test_data_loader = fine_tunning_trainer.get_eval_dataloader()
    test_logits = []
    test_attn = []
    for batch in test_data_loader:
         l, a = predict_with_mc(
             finetunned_ehr_former,
             batch,
             N_MC_SAMPLES
         )
         test_logits.append(l)
         test_attn.append(a)
    test_logits = torch.concat(test_logits).cpu().detach().numpy()
    test_attn = torch.concat(test_attn).cpu().detach().numpy()
    np.save(f'{OUTPUT}/mc_test_logits.npy', test_logits)
    np.save(f'{OUTPUT}/mc_test_attn.npy', test_attn)


parser = argparse.ArgumentParser()
parser.add_argument('--selected_category', type=str, required=True, help='The category to predict.')
parser.add_argument('--split_name', type=str, required=True, help='The split to select.')
parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pretrained EHRFormer.')
parser.add_argument('--task_name', type=str, required=True, help='Default task name.')
parser.add_argument('--epoch_nb', type=int, required=True, help='Default batch size for the model training.')
parser.add_argument('--run_nb', type=int, required=True, help='The number of a current run.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(
        selected_category=args.selected_category,
        split_name=args.split_name,
        run_nb=args.run_nb,
        pretrained_path=args.pretrained_path,
        task_name=args.task_name,
        epoch_nb=args.epoch_nb,
    )
