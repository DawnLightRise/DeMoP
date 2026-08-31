# Information for the Corresponding Manuscript:
# Tang, C., L. Yu, Q. Li, and L. Xu. 2026. DeMoP: A Language-Model-Guided Mixture-of-Experts
# Framework for Cancer Prognosis. bioRxiv preprint. https://doi.org/10.64898/2026.08.24.746579.
# Manuscript under consideration at Nature Machine Intelligence as of August 27, 2026.

# ============================================================
# 1. Environment and imports
# ============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from transformers import DebertaV2Tokenizer, DebertaV2Model, set_seed

# ============================================================
# 2. Configuration
# ============================================================

MODEL_DIR = "./deberta_finetuned_custom_OS_0422_EarlyStopping"
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "model_weights.pth")

INPUT_CSV = "G_data_llm_20250324.csv"
OUTPUT_CSV = "inference_predictions.csv"

TEXT_COLUMN = "description"
LABEL_COLUMN = "os_group"
BATCH_SIZE = 8
MAX_LEN = 512
NUM_EXPERTS = 4
NUM_CLASSES = 2

# ============================================================
# 3. Inference dataset
# ============================================================

class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, text_col="description", label_col=None, max_len=512):
        self.df = df.reset_index(drop=False)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        item = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "row_index": row["index"],
        }

        if self.label_col is not None and self.label_col in self.df.columns:
            item["labels"] = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        if "CANCER_TYPE" in self.df.columns:
            item["CANCER_TYPE"] = row["CANCER_TYPE"]
        if "CANCER_TYPE_DETAILED" in self.df.columns:
            item["CANCER_TYPE_DETAILED"] = row["CANCER_TYPE_DETAILED"]

        return item

# ============================================================
# 4. Model architecture
# ============================================================

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
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
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

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
        return self.resnet_classifier(x)


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
        expert_outputs = [expert(x) for expert in self.experts]   # [B, 2] * num_experts
        expert_outputs = torch.stack(expert_outputs, dim=1)       # [B, E, 2]
        gating_logits = self.gating(x)                            # [B, E]
        gating_weights = F.softmax(gating_logits, dim=-1)         # [B, E]
        gating_weights = gating_weights.unsqueeze(-1)             # [B, E, 1]
        output = torch.sum(gating_weights * expert_outputs, dim=1)  # [B, 2]
        return output


class CustomDebertaV3MoEForSequenceClassification(nn.Module):
    def __init__(self, num_experts=4):
        super(CustomDebertaV3MoEForSequenceClassification, self).__init__()
        self.deberta = DebertaV2Model.from_pretrained("microsoft/deberta-v3-large")
        self.mlp_reduce = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.GELU()
        )
        self.attention = Attention(1024)
        self.moe = MixtureOfExperts(hidden_size=1024, num_experts=num_experts, num_classes=2)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states
        concat_hidden = torch.cat(hidden_states[-4:], dim=-1)   # [B, L, 4096]
        reduced_hidden = self.mlp_reduce(concat_hidden)          # [B, L, 1024]
        pooled = self.attention(reduced_hidden)                  # [B, 1024]
        logits = self.moe(pooled)                                # [B, 2]
        return logits

# ============================================================
# 5. Single-text prediction helper
# ============================================================

def predict_single_text(text, tokenizer, model, device, max_len=512):
    model.eval()
    inputs = tokenizer.encode_plus(
        str(text),
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    return {
        "pred_label": pred,
        "prob_class0": float(probs[0]),
        "prob_class1": float(probs[1]),
    }

# ============================================================
# 6. Main
# ============================================================

def main():
    # --------------------------------------------------------
    # Step 1: Reproducibility and device setup
    # --------------------------------------------------------
    set_seed(42)
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------
    # Step 2: Load tokenizer, build model, and load weights
    # --------------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_DIR)

    print("Building model...")
    model = CustomDebertaV3MoEForSequenceClassification(num_experts=NUM_EXPERTS)

    print("Loading model weights...")
    state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Step 3: Load the input data and build the DataLoader
    # --------------------------------------------------------
    df = pd.read_csv(INPUT_CSV, index_col=0)
    dataset = InferenceDataset(
        df=df,
        tokenizer=tokenizer,
        text_col=TEXT_COLUMN,
        label_col=LABEL_COLUMN if LABEL_COLUMN in df.columns else None,
        max_len=MAX_LEN
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --------------------------------------------------------
    # Step 4: Run batch inference
    # --------------------------------------------------------
    all_row_indices = []
    all_logits = []
    all_labels = []
    all_cancer_types = []
    all_cancer_types_detailed = []

    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(logits, dim=-1)

            all_logits.append(probs.cpu().numpy())
            all_row_indices.extend(batch["row_index"].tolist())

            if "labels" in batch:
                all_labels.extend(batch["labels"].tolist())

            if "CANCER_TYPE" in batch:
                all_cancer_types.extend(batch["CANCER_TYPE"])
            if "CANCER_TYPE_DETAILED" in batch:
                all_cancer_types_detailed.extend(batch["CANCER_TYPE_DETAILED"])

    probs = np.concatenate(all_logits, axis=0)
    preds = np.argmax(probs, axis=-1)

    # --------------------------------------------------------
    # Step 5: Build and save the prediction table
    # --------------------------------------------------------
    results_df = pd.DataFrame({
        "row_index": all_row_indices,
        "pred_label": preds,
        "prob_class0": probs[:, 0],
        "prob_class1": probs[:, 1],
    })

    if len(all_labels) > 0:
        results_df["true_label"] = all_labels

    if len(all_cancer_types) == len(results_df):
        results_df["CANCER_TYPE"] = all_cancer_types

    if len(all_cancer_types_detailed) == len(results_df):
        results_df["CANCER_TYPE_DETAILED"] = all_cancer_types_detailed

    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved inference results to: {OUTPUT_CSV}")

    # --------------------------------------------------------
    # Step 6: Report inference metrics when labels are available
    # --------------------------------------------------------
    if len(all_labels) > 0:
        labels = np.array(all_labels)
        prob_class1 = probs[:, 1]

        acc = accuracy_score(labels, preds)
        f1_weighted = f1_score(labels, preds, average="weighted")
        f1_class1 = f1_score(labels, preds, pos_label=1)

        if len(np.unique(labels)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(labels, prob_class1)

        print("\n===== Inference Metrics =====")
        print(f"Accuracy:        {acc:.4f}")
        print(f"Weighted F1:     {f1_weighted:.4f}")
        print(f"Class 1 F1:      {f1_class1:.4f}")
        print(f"ROC AUC:         {auc:.4f}")

    # --------------------------------------------------------
    # Optional: single-text inference example
    # --------------------------------------------------------
    # sample_text = "Patient is a 65-year-old male with ..."
    # result = predict_single_text(sample_text, tokenizer, model, device, max_len=512)
    # print(result)

if __name__ == "__main__":
    main()
