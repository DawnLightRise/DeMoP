# DoMeRa

**DoMeRa: A DoRA-Adapted Language Model with a Mixture of Adaptively Routed Experts for Clinical Prognosis**

DoMeRa is a parameter-efficient language-model framework for clinical prognosis. It fine-tunes **DeBERTa-v3-large** with **DoRA** and combines it with an **attention-based sparse Mixture-of-Experts (MoE)** architecture to model heterogeneous clinical prediction patterns.

## Overview

DoMeRa is designed to retain the representation strength of a pretrained language model while substantially reducing the number of parameters that require fine-tuning. The framework introduces adaptive expert routing so that different samples can be processed by different specialized experts according to their learned clinical representations.

## Key Features

- **DoRA-based parameter-efficient fine-tuning** of DeBERTa-v3-large
- **Attention-based expert routing** for sample-specific expert selection
- **Shared and sparsely activated experts** for modeling common and heterogeneous clinical patterns
- **Top-2 adaptive routing** to select the most relevant experts for each sample
- **Load-balancing regularization** to reduce expert collapse and encourage effective expert utilization
- **Expert-routing and specialization analysis** across clinical subgroups

## Architecture

```text
Clinical Text
     |
     v
DeBERTa-v3-large
     |
   DoRA
     |
     v
Last-Layer Representation Fusion
     |
     v
Attention-Based Representation Learning
     |
     v
Adaptive Expert Router
     |
     +-------------------+
     |                   |
     v                   v
Shared Expert      Top-2 Routed Experts
                         |
                         v
                Sparse Mixture-of-Experts
                         |
                         v
                 Clinical Prognosis
```

## Research Focus

DoMeRa is intended for clinical prognosis tasks in which patient populations may exhibit substantial heterogeneity. In addition to predictive modeling, the framework supports analysis of expert utilization and routing patterns to investigate whether different experts specialize in distinct clinical subgroups.

## Model Components

- **Backbone:** DeBERTa-v3-large
- **Parameter-efficient adaptation:** DoRA
- **Routing:** Attention-based adaptive routing
- **Expert architecture:** Shared expert + sparse routed experts
- **Expert selection:** Top-2 routing
- **Regularization:** Load-balancing loss
- **Analysis:** Expert utilization and subgroup specialization
