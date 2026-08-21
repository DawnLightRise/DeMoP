import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2Model,
    TrainingArguments,
    get_scheduler,
    set_seed,
    DataCollatorWithPadding,
    AdamW,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState
)

set_seed(42)
torch.cuda.empty_cache()

data = pd.read_csv('G_data_llm_20250324.csv', index_col=0)
texts = data['description'].astype(str).tolist()
labels = data['os_group'].tolist()
cancer_types = data['CANCER_TYPE'].tolist()
cancer_types_detailed = data['CANCER_TYPE_DETAILED'].tolist()

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512, cancer_types=None, cancer_types_detailed=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cancer_types = cancer_types
        self.cancer_types_detailed = cancer_types_detailed

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        
        item = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        if self.cancer_types is not None:
            item['CANCER_TYPE'] = self.cancer_types[idx]
        if self.cancer_types_detailed is not None:
            item['CANCER_TYPE_DETAILED'] = self.cancer_types_detailed[idx]
        return item

dataset = TextDataset(
    texts, labels, tokenizer, max_len=512,
    cancer_types=cancer_types,
    cancer_types_detailed=cancer_types_detailed
)

index_to_position = {idx: pos for pos, idx in enumerate(data.index)}

train_indices_df = pd.read_csv('G_train_data_llm_20250324.csv', index_col=0)
test_indices_df = pd.read_csv('G_test_data_llm_20250324.csv', index_col=0)

train_indices = [index_to_position[idx] for idx in train_indices_df.index if idx in index_to_position]
test_indices = [index_to_position[idx] for idx in test_indices_df.index if idx in index_to_position]

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# NOTE ON `val_dataset`:
# Due to the limited size of the dataset and to allow a larger proportion of the
# available data to be used for model training, the entire `train_dataset` is used
# for training and the `val_dataset` below is intentionally defined as a subset of
# `train_dataset`, used only for training-control purposes such as early stopping
# and best-checkpoint selection. Thus, the `val_dataset` is not intended to provide
# an independent estimate of model generalization performance. Final model evaluation
# is performed on the `test_dataset`, which is independent of and non-overlapping with
# `train_dataset`.

train_size = len(train_dataset)
val_size = train_size // 2
generator = torch.Generator().manual_seed(42)
_, val_dataset = random_split(train_dataset, [train_size - val_size, val_size], generator=generator)

print(f"train_dataset: {len(train_dataset)}")
print(f"val_dataset: {len(val_dataset)}")
print(f"test_dataset: {len(test_dataset)}")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.input_dim)
        weights = self.softmax(scores)
        context = torch.matmul(weights, v)
        pooled = context.mean(dim=1)
        return pooled

class ResidualBlockFC(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(ResidualBlockFC, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + residual
        out = self.gelu(out)
        return out

class ResNetClassifier(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(ResNetClassifier, self).__init__()
        self.block1 = ResidualBlockFC(1024, 512, dropout_rate)
        self.block2 = ResidualBlockFC(512, 256, dropout_rate)
        self.fc_final = nn.Linear(256, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        logits = self.fc_final(x)
        return logits

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super(ExpertNetwork, self).__init__()
        self.resnet_classifier = ResNetClassifier(dropout_rate=0.1)
    
    def forward(self, x):
        logits = self.resnet_classifier(x)
        return logits

class MixtureOfExperts(nn.Module):
    def __init__(self, hidden_size=1024, num_experts=4, num_classes=2):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim=hidden_size, num_classes=num_classes)
            for _ in range(num_experts)
        ])
        self.gating = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_experts)
        )
    
    def forward(self, x):
        # x: [batch_size, hidden_size]
        expert_outputs = [expert(x) for expert in self.experts]   # [batch_size, num_classes]
        expert_outputs = torch.stack(expert_outputs, dim=1)       # [batch_size, num_experts, num_classes]
        gating_logits = self.gating(x)                            # [batch_size, num_experts]
        gating_weights = F.softmax(gating_logits, dim=-1)         # [batch_size, num_experts]
        gating_weights = gating_weights.unsqueeze(-1)             # [batch_size, num_experts, 1]
        output = torch.sum(gating_weights * expert_outputs, dim=1)  # [batch_size, num_classes]
        return output

class CustomDebertaV3MoEForSequenceClassification(nn.Module):
    def __init__(self, num_experts=4):
        super(CustomDebertaV3MoEForSequenceClassification, self).__init__()
        self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')
        self.mlp_reduce = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.GELU()
        )
        
        hidden_size = 1024
        self.attention = Attention(hidden_size)
        self.moe = MixtureOfExperts(hidden_size=hidden_size, num_experts=num_experts, num_classes=2)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states  # tuple of layers
        # [batch_size, seq_len, 4*1024=4096]
        concat_hidden = torch.cat(hidden_states[-4:], dim=-1)
        # 4096 -> 1024
        reduced_hidden = self.mlp_reduce(concat_hidden)
        # [batch_size, 1024]
        pooled = self.attention(reduced_hidden)
        # MoE logits
        logits = self.moe(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return {'loss': loss, 'logits': logits}

model = CustomDebertaV3MoEForSequenceClassification(num_experts=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir='./results_deberta_OS_0422_EarlyStopping',
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=10000,
    weight_decay=0.01,
    logging_dir='./logs_deberta',
    logging_steps=1000,
    eval_strategy="steps",
    eval_steps=10000,
    save_strategy="steps",
    save_steps=10000,
    learning_rate=5e-7,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    fp16=True
)

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

train_loader = DataLoader(
    train_dataset,
    batch_size=training_args.per_device_train_batch_size,
    shuffle=True,
    collate_fn=data_collator,
    drop_last=True
)

num_training_steps = training_args.num_train_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    prob_class1 = probs[:, 1]

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    if len(np.unique(labels)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(labels, prob_class1)

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels
        )
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=3, early_stopping_threshold=1.0e-6):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.prev_metric = None
        self.num_no_improvement = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        current_metric = metrics["eval_loss"]
        if self.prev_metric is None:
            self.prev_metric = current_metric
            self.num_no_improvement = 0
        else:
            if self.prev_metric - current_metric > self.early_stopping_threshold:
                self.num_no_improvement = 0
            else:
                self.num_no_improvement += 1
                print(f"[EarlyStopping] Count = {self.num_no_improvement} / {self.early_stopping_patience}")
                if self.num_no_improvement >= self.early_stopping_patience:
                    print("[EarlyStopping] Stopping training now.")
                    control.should_training_stop = True
        self.prev_metric = current_metric
        return control

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler)
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5,
    early_stopping_threshold=1e-6
)

trainer.add_callback(early_stopping_callback)

from transformers.trainer_utils import get_last_checkpoint

output_dir = training_args.output_dir
last_checkpoint = None

if os.path.isdir(output_dir):
    last_checkpoint = get_last_checkpoint(output_dir)

if last_checkpoint is not None:
    print(f"*** Found checkpoint {last_checkpoint}, resuming from there ***")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("*** No checkpoint found, starting from scratch ***")
    trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

test_results = trainer.evaluate(test_dataset)
print("\n======== FINAL TEST RESULTS ========\n")
print(f"Final Test results: {test_results}\n")

output_dir = './deberta_finetuned_custom_OS_0422_EarlyStopping'
os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir, 'model_weights.pth'))
torch.save(model, os.path.join(output_dir, 'full_model.pth'))
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

from collections import defaultdict

cancer_type_counts = defaultdict(int)
for i in range(len(test_dataset)):
    sample = test_dataset[i]
    ctype = sample["CANCER_TYPE"]
    cancer_type_counts[ctype] += 1

cancer_types_unique = list(cancer_type_counts.keys())
results_by_cancer_type = {}

for ctype in cancer_types_unique:
    indices = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        if sample["CANCER_TYPE"] == ctype:
            indices.append(i)
    if len(indices) < 200:
        continue
    type_subset = Subset(test_dataset, indices)
    type_result = trainer.evaluate(type_subset)
    results_by_cancer_type[ctype] = {"num_samples": len(indices), "metrics": type_result}
    
print("\n======== ALL CANCER TYPES' TEST RESULTS ========\n")
for ctype, result_dict in results_by_cancer_type.items():
    num_samples = result_dict["num_samples"]
    metrics = result_dict["metrics"]
    print(f"Cancer Type: {ctype} (n={num_samples}), Test Results: {metrics}\n")

cancer_type_detailed_counts = defaultdict(int)
for i in range(len(test_dataset)):
    sample = test_dataset[i]
    ctype_detailed = sample["CANCER_TYPE_DETAILED"]
    cancer_type_detailed_counts[ctype_detailed] += 1

cancer_types_detailed_unique = list(cancer_type_detailed_counts.keys())
results_by_cancer_type_detailed = {}

for ctype_detailed in cancer_types_detailed_unique:
    indices = []
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        if sample["CANCER_TYPE_DETAILED"] == ctype_detailed:
            indices.append(i)
    if len(indices) < 200:
        continue
    type_subset = Subset(test_dataset, indices)
    type_result = trainer.evaluate(type_subset)
    results_by_cancer_type_detailed[ctype_detailed] = {"num_samples": len(indices), "metrics": type_result}
    
print("\n======== ALL CANCER TYPE DETAILED TEST RESULTS ========\n")
for ctype_detailed, result_dict in results_by_cancer_type_detailed.items():
    num_samples = result_dict["num_samples"]
    metrics = result_dict["metrics"]
    print(f"Cancer Type Detailed: {ctype_detailed} (n={num_samples}), Test Results: {metrics}\n")
    
print("\n======== FINAL TEST RESULTS ========\n")
print(f"Final Test results: {test_results}\n")

pred_output = trainer.predict(test_dataset)

logits = pred_output.predictions
labels = pred_output.label_ids

# softmax -> probability
probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
preds = np.argmax(probs, axis=-1)
prob_class1 = probs[:, 1]

# Accuracy
acc_overall = accuracy_score(labels, preds)
print(f"\n=== Overall Accuracy: {acc_overall:.4f} ===\n")

# Class 1 F1
f1_class1_overall = f1_score(labels, preds, pos_label=1)
print(f"\n=== Overall Class 1 (3-year mortality) F1 Score: {f1_class1_overall:.4f} ===\n")

# AUC
if len(np.unique(labels)) < 2:
    auc_overall = np.nan
else:
    auc_overall = roc_auc_score(labels, prob_class1)
print(f"\n=== Overall ROC AUC: {auc_overall:.4f} ===\n")

cancer_type_list = []
cancer_type_detailed_list = []

for i in range(len(test_dataset)):
    sample = test_dataset[i]
    cancer_type_list.append(sample["CANCER_TYPE"])
    cancer_type_detailed_list.append(sample["CANCER_TYPE_DETAILED"])

results_df = pd.DataFrame({
    "true_label": labels,
    "pred_label": preds,
    "prob_class0": probs[:, 0],
    "prob_class1": probs[:, 1],
    "CANCER_TYPE": cancer_type_list,
    "CANCER_TYPE_DETAILED": cancer_type_detailed_list
})

results_df.to_csv("test_predictions_with_probability.csv", index=False)
print("Saved probabilities to test_predictions_with_probability.csv")

train_pred_output = trainer.predict(train_dataset)

train_logits = train_pred_output.predictions
train_labels = train_pred_output.label_ids

train_probs = torch.softmax(torch.tensor(train_logits), dim=-1).cpu().numpy()
train_preds = np.argmax(train_probs, axis=-1)

train_cancer_type_list = []
train_cancer_type_detailed_list = []

for i in range(len(train_dataset)):
    sample = train_dataset[i]
    train_cancer_type_list.append(sample["CANCER_TYPE"])
    train_cancer_type_detailed_list.append(sample["CANCER_TYPE_DETAILED"])

train_results_df = pd.DataFrame({
    "true_label": train_labels,
    "pred_label": train_preds,
    "prob_class0": train_probs[:, 0],
    "prob_class1": train_probs[:, 1],
    "CANCER_TYPE": train_cancer_type_list,
    "CANCER_TYPE_DETAILED": train_cancer_type_detailed_list
})

train_results_df.to_csv("train_predictions_with_probability.csv", index=False)
print("Saved probabilities to train_predictions_with_probability.csv")

val_pred_output = trainer.predict(val_dataset)

val_logits = val_pred_output.predictions
val_labels = val_pred_output.label_ids

val_probs = torch.softmax(torch.tensor(val_logits), dim=-1).cpu().numpy()
val_preds = np.argmax(val_probs, axis=-1)

val_cancer_type_list = []
val_cancer_type_detailed_list = []

for i in range(len(val_dataset)):
    sample = val_dataset[i]
    val_cancer_type_list.append(sample["CANCER_TYPE"])
    val_cancer_type_detailed_list.append(sample["CANCER_TYPE_DETAILED"])

val_results_df = pd.DataFrame({
    "true_label": val_labels,
    "pred_label": val_preds,
    "prob_class0": val_probs[:, 0],
    "prob_class1": val_probs[:, 1],
    "CANCER_TYPE": val_cancer_type_list,
    "CANCER_TYPE_DETAILED": val_cancer_type_detailed_list
})

val_results_df.to_csv("val_predictions_with_probability.csv", index=False)
print("Saved probabilities to val_predictions_with_probability.csv")

type_preds = defaultdict(list)
type_labels = defaultdict(list)
type_probs = defaultdict(list)

for idx in range(len(test_dataset)):
    sample = test_dataset[idx]
    ctype = sample["CANCER_TYPE"]
    type_preds[ctype].append(preds[idx])
    type_labels[ctype].append(labels[idx])
    type_probs[ctype].append(prob_class1[idx])

print("=== Accuracy / Class 1 F1 / ROC AUC by CANCER_TYPE (n>=200) ===")
for ctype, labs in type_labels.items():
    if len(labs) < 200:
        continue

    acc_ct = accuracy_score(labs, type_preds[ctype])
    f1_ct = f1_score(labs, type_preds[ctype], pos_label=1)

    if len(set(labs)) < 2:
        auc_ct = np.nan
        print(f"{ctype} (n={len(labs)}): Accuracy = {acc_ct:.4f}, Class 1 F1 = {f1_ct:.4f}, ROC AUC = nan (only one class present)")
    else:
        auc_ct = roc_auc_score(labs, type_probs[ctype])
        print(f"{ctype} (n={len(labs)}): Accuracy = {acc_ct:.4f}, Class 1 F1 = {f1_ct:.4f}, ROC AUC = {auc_ct:.4f}")
print()

detailed_preds = defaultdict(list)
detailed_labels = defaultdict(list)
detailed_probs = defaultdict(list)

for idx in range(len(test_dataset)):
    sample = test_dataset[idx]
    ctype_det = sample["CANCER_TYPE_DETAILED"]
    detailed_preds[ctype_det].append(preds[idx])
    detailed_labels[ctype_det].append(labels[idx])
    detailed_probs[ctype_det].append(prob_class1[idx])

print("=== Accuracy / Class 1 F1 / ROC AUC by CANCER_TYPE_DETAILED (n>=200) ===")

for ctype_det, labs in detailed_labels.items():
    if len(labs) < 200:
        continue

    acc_det = accuracy_score(labs, detailed_preds[ctype_det])
    f1_det = f1_score(labs, detailed_preds[ctype_det], pos_label=1)

    if len(set(labs)) < 2:
        auc_det = np.nan
        print(f"{ctype_det} (n={len(labs)}): Accuracy = {acc_det:.4f}, Class 1 F1 = {f1_det:.4f}, ROC AUC = nan (only one class present)")
    else:
        auc_det = roc_auc_score(labs, detailed_probs[ctype_det])
        print(f"{ctype_det} (n={len(labs)}): Accuracy = {acc_det:.4f}, Class 1 F1 = {f1_det:.4f}, ROC AUC = {auc_det:.4f}")
