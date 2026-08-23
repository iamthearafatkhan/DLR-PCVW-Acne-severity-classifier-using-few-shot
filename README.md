<div align="center">

# 🧬 DLR-PCVW

### Distribution-Aware Few-Shot Learning for Acne Severity Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://acnegrad-ai.streamlit.app)
[![Hugging Face](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/iamthearafatkhan/dlr-pcvw-acne-severity)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/iamthearafatkhan/DLR-PCVW-Acne-severity-classifier-using-few-shot)](https://github.com/iamthearafatkhan/DLR-PCVW-Acne-severity-classifier-using-few-shot)

**🎓 Research Prototype · Thesis Demonstration**

</div>

---

## 📋 Overview

**DLR-PCVW** is a **Few-Shot Learning** framework designed for classifying acne severity into 4 grades. It leverages a **Distribution-aware Lesion Representation** with **Prototype-based Classification** to achieve robust performance with limited data — making it ideal for medical imaging applications where labeled data is scarce.

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **4-Way Classification** | Clear, Mild, Moderate, Severe |
| **Few-Shot Learning** | 1, 3, 5, or 10-shot inference |
| **Ensemble Model** | 5 seeds for robust predictions |
| **Attention Visualization** | Lesion focus maps for interpretability |
| **Prototype Similarity** | Compare with class prototypes |
| **Interactive UI** | Drag-and-drop image upload |
| **Dark Theme** | Modern glassmorphism UI |

---

## 🏗️ Architecture

<div align="center">
  <img src="image/proposed_architecture.png" alt="DLR-PCVW Architecture" width="900"/>
  <br>
  <em>DLR-PCVW: Distribution-aware Lesion Representation with Prototype-based Few-Shot Classification</em>
</div>

### Pipeline Overview

| Step | Component | Description |
|:----:|-----------|-------------|
| **1** | 📥 Input | 224×224 RGB Image |
| **2** | 🔍 Feature Extractor | EfficientNet-B0 (Pre-trained) |
| **3** | 🎯 FiLM Calibration | Feature-wise Linear Modulation |
| **4** | 👁️ Lesion Attention | Soft Attention Map for Lesions |
| **5** | 📊 PCVW Representation | μ + w·σ (Variance Weighted) |
| **6** | 🧩 Prototype Network | Cosine Similarity to Prototypes |
| **7** | ✅ Output | 4-Way Severity Grades |

### Model Specifications

| Component | Details |
|-----------|---------|
| **Framework** | DLR-PCVW |
| **Backbone** | EfficientNet-B0 (Pre-trained) |
| **Learning** | Few-Shot Learning |
| **Task** | 4-Way Classification |
| **Input** | 224 × 224 |
| **Trainable Parameters** | 6.93M |
| **Ensemble** | 5 Random Seeds |

---

## 📊 Model Performance

*Results averaged across 5 random seeds:*

| Shot | Accuracy | Confidence Interval |
|------|----------|---------------------|
| **1-Shot** | 76.65% | ± 1.00% |
| **3-Shot** | 79.40% | ± 0.73% |
| **5-Shot** | 79.60% | ± 0.62% |
| **10-Shot** | 80.27% | ± 0.43% |


This is internal performance and showing feature ectivations
<div align="center">
  <img src="image/feature_activations.png" alt="Model Performance" width="600"/>
</div>



---

## 🔬 Technical Details

### Feature Extraction

**EfficientNet-B0** is used as the backbone network to extract a **1280-dimensional feature representation** from each input image.

### FiLM Calibration

Feature-wise Linear Modulation (FiLM) is used to adapt extracted features for the target few-shot classification task.

The calibration mechanism learns feature-specific modulation parameters to improve task adaptation.

### Lesion Attention

A soft attention mechanism is used to emphasize regions associated with acne lesions while reducing the influence of irrelevant background information.

### PCVW Representation

The framework represents each feature using a combination of its mean and standard deviation:

[
z = \mu + w \cdot \sigma
]

where:

* (\mu) = feature mean
* (\sigma) = feature standard deviation
* (w) = learnable variance-weighting parameter

This allows the model to incorporate both central feature information and feature variability.

### Prototype Network

Class prototypes are constructed from the support examples.

Classification is performed using **cosine similarity** between the query representation and class prototypes.

---

## ⚙️ Training Configuration

| Parameter         |            Value |
| ----------------- | ---------------: |
| Training Episodes |            1,200 |
| Testing Episodes  |              300 |
| Backbone          |  EfficientNet-B0 |
| Feature Dimension |             1280 |
| Optimizer         |            AdamW |
| Learning Rate     |         2 × 10⁻⁴ |
| Weight Decay      |            0.008 |
| Scheduler         | Cosine Annealing |
| Label Smoothing   |             0.05 |
| Temperature       |             16.0 |

---

## 📉 Loss Functions

The training objective combines multiple losses:

### Cross-Entropy Loss

Standard classification loss used to optimize the predicted class probabilities.

### Label Distribution Learning (LDL) Loss

Encourages the model to learn the distributional characteristics of acne severity labels rather than relying only on hard class assignments.

### Multi-Head Consistency Loss

Encourages predictions from different model heads to remain consistent, improving prediction stability and representation quality.

---

## 📂 Project Structure

```text
DLR-PCVW-Acne-severity-classifier-using-few-shot/
│
├── app.py
│   └── Main Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── pyproject.toml
│   └── Project configuration
│
├── create_support_set.py
│   └── Support-set generation utility
│
├── README.md
│   └── Project documentation
│
├── image/
│   └── proposed_architecture.png
│       └── Proposed DLR-PCVW architecture
│
├── .streamlit/
│   └── config.toml
│       └── Streamlit configuration
│
├── saved_models/
│   └── ensemble_models.pt
│       └── Automatically downloaded ensemble model
│
└── support_set/
    └── support_set.pt
        └── Automatically downloaded support set
```

---



## 🧪 Few-Shot Learning

Unlike conventional deep-learning approaches that require large numbers of labeled examples for every class, this framework performs classification using a limited number of support examples.

The support set provides representative examples for each acne severity class.

For a query image:

1. Extract the feature representation.
2. Apply feature calibration and lesion-aware processing.
3. Generate the PCVW representation.
4. Compare the query representation with class prototypes.
5. Calculate cosine similarity.
6. Predict the most similar acne severity class.

---

## 🎯 Key Contributions

The DLR-PCVW framework incorporates several components specifically designed for few-shot acne severity classification:

* Lesion-aware feature representation
* FiLM-based task-specific feature calibration
* Per-Channel Variance Weighting
* Prototype-based few-shot classification
* Label Distribution Learning
* Multi-head consistency regularization
* Ensemble-based prediction
* Streamlit deployment for practical inference

---

## 🛠️ Technologies

| Technology      | Purpose                 |
| --------------- | ----------------------- |
| Python          | Core implementation     |
| PyTorch         | Deep learning framework |
| EfficientNet-B0 | Feature extraction      |
| Streamlit       | Web application         |
| NumPy           | Numerical computation   |
| Pillow          | Image processing        |
| Hugging Face    | Model hosting           |

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.

The predictions generated by this system should **not be considered a medical diagnosis or a substitute for professional dermatological assessment**.

---


## 📄 License

This project is part of a **thesis research** and is subject to academic copyright.

**© 2026 [Your Full Name]. All Rights Reserved.**

### Terms of Use

This code is made available for:

| ✅ Permitted | ❌ Prohibited |
|--------------|---------------|
| Academic review and evaluation | Commercial use |
| Thesis examination and defense | Redistribution without permission |
| Citation in academic papers | Modification without permission |
| Reference in future research | Claiming as your own work |

### How to Cite

If you reference this work in academic context, please cite:

Md Arafat Hossen Raaby. (2026). DLR-PCVW: A Lesion-Aware Distribution-Weighted
Framework for Few-Shot Dermatological Severity Assessment. Thesis. 
Premier University Chittagong, Department of Computer Science 
and Engineering.
