# Chain-of-Affect_Compound-FER
This is the repository for the Chain of Affect model, a VLM-based pipeline for compound FER

## 📑 Table of Contents
- [About the Project](#about-the-project)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Finetuning](#finetuning)
- [Inference](#inference)
- [Webcam Demo (Proof of Concept)](#webcam-demo-proof-of-concept)
- [Pretrained Model (Hugging Face)](#pretrained-model-hugging-face)
- [References and Acknowledgements](#references-and-acknowledgements)
- [License](#license)
- [Citation](#citation)

---

## 🧠 About the Project

This repository accompanies the paper:

> **Chain-of-Affect: Compound Facial Expression Recognition through Sequential Vision Language Model Prompting for Interactive Robots**  
> (Submitted to EAAI 202X)

It contains:
- Code to **finetune** [Phi-3.5 Vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) on the RAF-DB training set using the proposed Chain-of-Affect prompting method.
- Scripts for **inference** on the RAF-DB test set for both basic and compound facial expression recognition (FER).
- A **webcam demo** script to test real-time compound FER.

---

## ⚙️ Installation

Set up the environment using `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate phi3v
```

---

## 📂 Dataset Preparation

We use the RAF-DB dataset for both training and evaluation.
1. Visit the RAF-DB dataset page and request access:
http://www.whdeng.cn/RAF/model1.html
2. After 
