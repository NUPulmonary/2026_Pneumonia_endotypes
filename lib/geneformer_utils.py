import numpy as np
import scipy.sparse
import sklearn.metrics
import transformers
import torch

import common_data


def freeze_encoder_layers(
    model,
    layers=4,
    encoder_layer_prefix='bert.encoder.layer.',
    unfreeze=False,
):
    layers = layers if isinstance(layers, list) else list(range(layers))
    for tensor_name, tensor_ in model.state_dict().items():
        if not tensor_name.startswith(encoder_layer_prefix):
            continue
        tensor_name_without_prefix = tensor_name[len(encoder_layer_prefix):]
        layer_int = int(tensor_name_without_prefix.split('.')[0])
        if layer_int in layers:
            tensor_.requires_grad = unfreeze


def freeze_model_embeddings(model, unfreeze=False):
    for tensor_name, tensor_ in model.state_dict().items():
        if tensor_name.startswith('bert.embeddings.'):
            tensor_.requires_grad = unfreeze


def get_model(
        path: str = common_data.GENEFORMER_ROOT,
        num_labels: int = 2,
        freeze_layers: int = 0,
        freeze_embeddings: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        enable_random_dropout: bool = False,
        move_to: str = 'cuda',
    ):
    model = transformers.BertForSequenceClassification.from_pretrained(
        path,
        num_labels=num_labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    larger_dropout_config_dict = model.config
    # changed from 0.02
    larger_dropout_config_dict.attention_probs_dropout_prob = 0.2
    larger_dropout_config_dict.hidden_dropout_prob = 0.2
    larger_dropout_model = transformers.BertForSequenceClassification.from_pretrained(
        path,
        config=larger_dropout_config_dict
    )
    model = larger_dropout_model

    if freeze_layers > 0:
        freeze_encoder_layers(model, freeze_layers)

    if freeze_embeddings:
        freeze_model_embeddings(model)

    if enable_random_dropout:
        model.train()

    return model.to(move_to)


def get_training_args(**kwargs):
    defaults = {
        "learning_rate": 2e-5,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": 5,
        "group_by_length": False,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": 'linear',
        "warmup_ratio": 0.25,
        "weight_decay": 0.0001, #changed (0.001)
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "load_best_model_at_end": True,
        "gradient_accumulation_steps": 3, # Added
        "report_to": "wandb",
    }
    defaults.update(kwargs)
    return defaults


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_probs = scipy.special.softmax(logits, axis=1)
    y_pred = np.argmax(logits, axis=-1)

    auprc = sklearn.metrics.average_precision_score(y_true, y_probs[:, 1])
    f1_binary = sklearn.metrics.f1_score(y_true, y_pred, average='binary')
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_probs[:, 1])
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


### Compute attention weights for genes
def aggregate_attention_weights(attention_weights_, attention_of_indices=0):
    final_indices = (
        attention_of_indices
        if isinstance(attention_of_indices, list)
        else [attention_of_indices]
    )
    acc = attention_weights_[-1][:, final_indices, :]
    for att_matrix in attention_weights_[:-1][::-1]:
        acc = torch.matmul(acc, att_matrix)
    return acc


def compute_attention_weights_for_ids(
        ids,
        attention_mask,
        attentions,
        attention_of_indices=0,
):
    attentions_avg_over_heads = [att_.mean(axis=1) for att_ in attentions]
    attention_weights = aggregate_attention_weights(
        attentions_avg_over_heads,
        attention_of_indices
    )
    return attention_weights


def forward_and_return_attn(
        model,
        batch,
        total_genes: int,
        multiply_by_n_tokens: bool = False,
):
    gene_ids, attention_mask = batch['input_ids'], batch['attention_mask']
    model_predictions = model(
        input_ids=gene_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=False,
    )

    attention_weights = compute_attention_weights_for_ids(
        gene_ids,
        attention_mask,
        model_predictions['attentions'],
    )
    result_attn = []
    result_rank = []
    result_logits = []
    for genes, cell_attention_mask, cell_attention_weight, cell_logits in zip(
        gene_ids.cpu().detach().numpy(),
        attention_mask.cpu().detach().numpy(),
        attention_weights[:, 0].cpu().detach().numpy(),
        model_predictions['logits'].cpu().detach().numpy(),
    ):
        n_genes = (genes > 0).sum()

        if multiply_by_n_tokens:
            cell_attention_weight *= n_genes
        r_i = [0] * n_genes
        c_i = genes[genes > 0]
        # sometimes genes and mask is at truncated dimension when no cell in batch has enough genes
        data = cell_attention_weight[:genes.shape[0]][genes > 0]

        attn_matrix = scipy.sparse.coo_matrix((data, (r_i, c_i)), shape=(1, total_genes))
        rank_matrix = scipy.sparse.coo_matrix((list(range(n_genes)), (r_i, c_i)), shape=(1, total_genes))

        result_attn.append(attn_matrix)
        result_rank.append(rank_matrix)
        result_logits.append(cell_logits)

    return result_attn, result_rank, result_logits
