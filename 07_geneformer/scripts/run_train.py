import sys
sys.path.insert(0, '../lib')

import glob
import os
import pickle

import datasets
import geneformer
import numpy as np
import pandas as pd
import transformers
import wandb

import common_data
import geneformer_utils


# 'python scripts/run_train.py "{input.tokens}" "{wildcards.task}" "{wildcards.split}" "{output}"'
TOKENS = sys.argv[1]
TASK = sys.argv[2]
SPLIT = sys.argv[3]
OUTPUT = sys.argv[4]


def load_data(token_path):
    # Filter files with prefix "chunk"
    chunks = glob.glob(f'{token_path}/chunk*.dataset')
    tokenized_chunks = [datasets.load_from_disk(file) for file in chunks]
    return datasets.concatenate_datasets(tokenized_chunks)


def filter_split_data(data, task: common_data.TaskInfo):
    task_column = task
    idx = np.isin(data[task.column], task.column_values)
    train_idx = idx & np.char.equal(data[task.split_column], 'train')
    test_idx = idx & np.char.equal(data[task.split_column], 'test')
    data = data.rename_column(task.column, 'label')

    train = data.select(np.where(train_idx)[0])
    test = data.select(np.where(test_idx)[0])

    if task.column_values != [0, 1]:
        values_to_labels = {v: i for i, v in enumerate(task.column_values)}
        train = train.map(lambda x: {'label': values_to_labels[x['label']]}, num_proc=16)
        test = test.map(lambda x: {'label': values_to_labels[x['label']]}, num_proc=16)

    return train, test


def downsample(dataset, rng):
    control_idx = np.where(np.equal(dataset['label'], 0))[0]
    perturbation_idx = np.where(np.equal(dataset['label'], 1))[0]

    if len(control_idx) > len(perturbation_idx):
        smaller_idx = perturbation_idx
        larger_idx = control_idx
    else:
        smaller_idx = control_idx
        larger_idx = perturbation_idx

    final_idx = np.concatenate([
        smaller_idx,
        rng.choice(
            larger_idx,
            size=len(smaller_idx),
            replace=False
        )
    ])
    return dataset.select(final_idx)


def init_wandb(task: common_data.TaskInfo, params, model, train_data, eval_data):
    hyperparameters = params.copy()
    hyperparameters['freeze_layers'] = 4
    hyperparameters['freeze_embeddings'] = True
    hyperparameters['architecture'] = 'Geneformer_6l'
    hyperparameters['train/n_cells_class_0'] = np.equal(train_data['label'], 0).sum()
    hyperparameters['train/n_cells_class_1'] = np.equal(train_data['label'], 1).sum()
    hyperparameters['eval/n_cells_class_0'] = np.equal(eval_data['label'], 0).sum()
    hyperparameters['eval/n_cells_class_1'] = np.equal(eval_data['label'], 1).sum()
    hyperparameters.update(model.config.to_diff_dict())

    group_name = task.column
    if task.pathname not in common_data.TASKS:
        group_name = f'{task.column_values[0]}_vs_{task.column_values[1]} v2'
    model_name = f'{group_name}_{SPLIT}_{os.path.basename(OUTPUT).replace("model_", "")}'
    wandb.init(
        project="serniczek",
        group=group_name,
        name=model_name,
        config=hyperparameters,
    )


rng = np.random.default_rng(1066)
task = common_data.get_task_info(TASK, SPLIT)
data = load_data(TOKENS)
train, test = filter_split_data(data, task)
train = downsample(train, rng)

print(f'Train size: {len(train)}')
print(f'Test size: {len(test)}')

model = geneformer_utils.get_model(freeze_layers=4)
training_args_dict = geneformer_utils.get_training_args(
    output_dir=OUTPUT,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=8,
    max_grad_norm=0.1,
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
)

init_wandb(task, training_args_dict, model, train, test)

training_args = transformers.training_args.TrainingArguments(**training_args_dict)
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    data_collator=geneformer.DataCollatorForCellClassification(),
    train_dataset=train,
    eval_dataset=test,
    compute_metrics=geneformer_utils.compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT)

wandb.finish()

# predictions = trainer.predict(test)
# with open(f'{OUTPUT}/predictions_test.pkl', 'wb') as fp:
#     pickle.dump(predictions, fp)
# trainer.save_metrics('eval_test', predictions.metrics)

# predictions_train = trainer.predict(train)
# with open(f'{OUTPUT}/predictions_train_downsampled.pkl', 'wb') as fp:
#     pickle.dump(predictions, fp)
# trainer.save_metrics('eval_train_downsampled', predictions.metrics)
