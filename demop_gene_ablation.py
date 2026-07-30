import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV2Tokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import re

# ------------------ 1. 固定随机种子 & 清理缓存 ------------------
torch.manual_seed(42)
torch.cuda.empty_cache()

# ------------------ 2. 初始化分词器 (新 finetuned) ------------------
tokenizer = DebertaV2Tokenizer.from_pretrained('./deberta_finetuned_custom_OS_0422_EarlyStopping')

# ------------------ 3. 自定义 Dataset ------------------
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

# ------------------ 4. 定义 Attention ------------------
class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key   = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, L, D]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.input_dim)
        weights = self.softmax(scores)
        context = torch.matmul(weights, v)
        pooled = context.mean(dim=1)  # [B, D]
        return pooled

# ------------------ 5. ResNet-style Residual Block & Expert ------------------
class ResidualBlockFC(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_features, out_features)
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.gelu(out + res)

class ResNetClassifier(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.block1 = ResidualBlockFC(1024, 512, dropout_rate)
        self.block2 = ResidualBlockFC(512, 256, dropout_rate)
        self.fc_final = nn.Linear(256, 2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.fc_final(x)

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super().__init__()
        self.resnet_classifier = ResNetClassifier(dropout_rate=0.1)

    def forward(self, x):
        return self.resnet_classifier(x)

# ------------------ 6. Mixture of Experts 模块 ------------------
class MixtureOfExperts(nn.Module):
    def __init__(self, hidden_size=1024, num_experts=4, num_classes=2):
        super().__init__()
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
        # x: [B, D]
        expert_logits = torch.stack([e(x) for e in self.experts], dim=1)      # [B, E, C]
        gate_logits = self.gating(x)                                          # [B, E]
        gate_w = F.softmax(gate_logits, dim=-1).unsqueeze(-1)                 # [B, E, 1]
        return torch.sum(gate_w * expert_logits, dim=1)                       # [B, C]

# ------------------ 7. 整体模型 ------------------
class CustomDebertaV3MoEForSequenceClassification(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        from transformers import DebertaV2Model
        self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large')
        # 从最后4层拼接后直接降维 4096->1024
        self.mlp_reduce = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.GELU()
        )
        self.attention = Attention(1024)
        self.moe       = MixtureOfExperts(hidden_size=1024, num_experts=num_experts, num_classes=2)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hs = outputs.hidden_states
        x = torch.cat(hs[-4:], dim=-1)       # [B, L, 4*1024]
        x = self.mlp_reduce(x)               # [B, L, 1024]
        x = self.attention(x)                # [B, 1024]
        logits = self.moe(x)                 # [B, 2]
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'loss': loss, 'logits': logits}

# ------------------ 8. 加载模型 ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomDebertaV3MoEForSequenceClassification(num_experts=4)
state_dict = torch.load(
    './deberta_finetuned_custom_OS_0422_EarlyStopping/model_weights.pth',
    map_location=device
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ------------------ 9. 构建 DataCollator & Trainer ------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted')
    }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=TrainingArguments(
        output_dir='./tmp_eval',
        per_device_eval_batch_size=8,
        fp16=True
    ),
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ------------------ 10. 读取数据 & 构造 Dataset ------------------
data_all = pd.read_csv('G_data_llm_20250324.csv', index_col=0)
texts_all = data_all['description'].astype(str).tolist()
labels_all = data_all['os_group'].tolist()
cancer_types_all = data_all['CANCER_TYPE'].tolist()
cancer_types_detailed_all = data_all['CANCER_TYPE_DETAILED'].tolist()

dataset_all = TextDataset(
    texts_all, labels_all, tokenizer,
    max_len=512,
    cancer_types=cancer_types_all,
    cancer_types_detailed=cancer_types_detailed_all
)

# ------------------ 11. 辅助函数：预测概率 ------------------
def get_prediction_prob(text, tokenizer, model, device):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors="pt"
    )
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs['logits']
    probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()[0]

# ------------------ 12. 辅助函数：按括号外分号切分 ------------------
def split_by_semicolon_outside_parentheses(text):
    segments = []
    current = []
    depth = 0
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth = max(depth-1, 0)
        if char == ';' and depth == 0:
            seg = ''.join(current).strip()
            if seg:
                segments.append(seg)
            current = []
        else:
            current.append(char)
    if current:
        seg = ''.join(current).strip()
        if seg:
            segments.append(seg)
    return segments

# ------------------ 13. 辅助函数：提取基因单元 ------------------
def extract_gene_segments(text):
    m = re.search(r"GENE MUTATION PROFILE:(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    profile = m.group(1).strip()
    segments = split_by_semicolon_outside_parentheses(profile)
    return [seg for seg in segments if seg]

# ------------------ 14. 辅助函数：删除单个基因单元 ------------------
def remove_gene_segment(text, seg_to_remove):
    m = re.search(r"(.*GENE MUTATION PROFILE:)(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return text
    prefix, profile = m.group(1), m.group(2)
    # 删除目标片段及可能的分号和空格
    pattern = re.escape(seg_to_remove) + r"\s*;?"
    new_profile = re.sub(pattern, "", profile, flags=re.IGNORECASE).strip()
    # 如果删完没剩，去掉 PROFILE:
    if not new_profile:
        return prefix
    return prefix + " " + new_profile

# ------------------ 15. 辅助函数：逐段消融 ------------------
def ablate_text(text, tokenizer, model, device):
    # 原始预测
    orig_probs = get_prediction_prob(text, tokenizer, model, device)
    orig_pred = np.argmax(orig_probs)
    # 提取所有基因单元
    segs = extract_gene_segments(text)
    results = {}
    for seg in segs:
        # 删除该单元后再预测
        txt2 = remove_gene_segment(text, seg)
        probs2 = get_prediction_prob(txt2, tokenizer, model, device)
        # 计算原类别概率的下降值
        delta = orig_probs[orig_pred] - probs2[orig_pred]
        results[seg] = delta
    return orig_pred, orig_probs[orig_pred], results

# ------------------ 16. 全样本消融统计（带进度打印） ------------------
from collections import defaultdict

gene_delta_dict = defaultdict(list)
data_all = pd.read_csv('G_data_llm_20250324.csv', index_col=0)
texts_all = data_all['description'].astype(str).tolist()
total = len(texts_all)

for i, txt in enumerate(texts_all):
    # 每条都打印当前进度
    print(f"Processing sample {i+1}/{total}")
    _, _, ablation = ablate_text(txt, tokenizer, model, device)
    for seg, d in ablation.items():
        gene_delta_dict[seg].append(d)
        # 每 100 条再打印一次该片段的 Δ 值
        if i % 100 == 0:
            print(f"  Seg: '{seg}' -> delta: {d:.6f}")
    if i % 100 == 0:
        print("\n")

# ------------------ 17. 平均 Δ 值排序 ------------------
avg_delta = {seg: np.mean(ds) for seg, ds in gene_delta_dict.items()}
sorted_avg = sorted(avg_delta.items(), key=lambda x: x[1], reverse=True)

print("\n====== 含增益/丢失 注释的基因 Δ 排名 ======")
for seg, val in sorted_avg:
    print(f"Seg: '{seg}' -> Avg delta: {val:.6f}")

# ------------------ 18. 按片段前 20 字分组统计（带分组进度） ------------------
grouped = defaultdict(list)
for seg, ds in gene_delta_dict.items():
    key = seg[:20]
    grouped[key].extend(ds)
avg_grouped = {k: np.mean(v) for k, v in grouped.items()}
sorted_grouped = sorted(avg_grouped.items(), key=lambda x: x[1], reverse=True)

print("\n====== 前20字符分组的基因 Δ 排名 ======")
for grp, val in sorted_grouped:
    print(f"Group: '{grp}' -> Avg delta: {val:.6f}")

