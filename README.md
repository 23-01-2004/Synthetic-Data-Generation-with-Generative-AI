# Synthetic Data Generation with Generative AI

A comprehensive 5-phase project demonstrating advanced techniques for handling class imbalance using Generative AI.

<p align="center"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" height="25"> <img src="https://img.shields.io/badge/Visualization-FF6B6B?style=for-the-badge&logo=matplotlib&logoColor=white" height="25"> <img src="https://img.shields.io/badge/Machine%20Learning-4ECDC4?style=for-the-badge&logo=scikit-learn&logoColor=white" height="25"> <img src="https://img.shields.io/badge/Deep%20Learning-45B7D1?style=for-the-badge&logo=pytorch&logoColor=white" height="25"> <img src="https://img.shields.io/badge/Utilities-96CEB4?style=for-the-badge&logo=jupyter&logoColor=white" height="25"> </p>

**Project Overview**

This project addresses the critical challenge of class imbalance in machine learning, particularly in high-stakes domains like fraud detection and medical diagnosis.
Traditional models often fail to identify rare but crucial cases.
We implemented Generative AI techniques (GANs and VAEs) to create realistic synthetic data for the minority class, improving model performance significantly.

 Table of Contents

Project Overview

What We Accomplished

Tech Stack

Project Structure

Installation & Setup

Project Phases

Key Results

Getting Started

Documentation

Contributing



 The Problem

Severe class imbalance (often 99:1 ratio in fraud detection)

Standard models biased toward majority class

Poor detection of rare but critical cases

Business impact: Missed frauds, undiagnosed diseases, security breaches

 Our Solution

Generative Adversarial Networks (GANs) for tabular data

Variational Autoencoders (VAEs) for robust synthetic generation

Comprehensive validation with statistical tests and visualizations

Business impact analysis with cost-benefit evaluation

 What We Accomplished
Phase 1: Foundation & Data Preparation

Established project structure and environment

Analyzed dataset imbalance and defined success metrics

Created baseline model

Phase 2: Comprehensive EDA

Statistical analysis of class imbalance

Feature importance and correlation visualization

Dimensionality reduction (PCA, t-SNE)

Outlier detection and data quality assessment

Phase 3: Synthetic Data Generation

Implemented CTGAN (Conditional Tabular GAN)

Implemented VAE (Variational Autoencoder)

Compared with SMOTE

Validated data quality statistically

Phase 4: Model Training & Evaluation

Trained XGBoost, Random Forest, Logistic Regression

Compared models across all datasets

Performed statistical significance testing

Conducted business impact evaluation

Phase 5: Documentation & Deployment

Detailed documentation and visual summaries

Deployment roadmap and monitoring framework

Reproducibility and scalability ensured

 Tech Stack
 Core Data Science
import pandas as pd
import numpy as np
import scikit-learn as sklearn
import scipy as sp

 Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

 Machine Learning
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

 Deep Learning & Generative AI
import torch
import tensorflow as tf
from ctgan import CTGAN
from sdv.tabular import CTGAN as SDV_CTGAN
from sdv.tabular import TVAE

 Utilities
import jupyter
import notebook
import tqdm
import pyyaml
import json
import warnings

 Project Structure
synthetic_data_imbalanced/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and preprocessed data
│   ├── synthetic/              # Generated synthetic data
│   └── augmented/              # Augmented training sets
├── notebooks/                  
│   ├── 01_eda_baseline.ipynb
│   ├── 02_comprehensive_eda.ipynb
│   ├── 03_synthetic_data_generation.ipynb
│   ├── 04_model_training_evaluation.ipynb
│   └── 05_documentation_deployment.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── synthetic_generation.py
│   ├── model_training.py
│   └── evaluation.py
├── config/
│   └── parameters.yaml
├── results/
│   ├── eda/
│   ├── synthetic_data/
│   ├── model_evaluation/
│   └── final_deliverables/
├── requirements.txt
└── README.md

🔧 Installation & Setup
# 1. Clone Repository
git clone https://github.com/your-username/synthetic-data-imbalanced.git
cd synthetic-data-imbalanced

# 2. Create Virtual Environment
python -m venv synthetic_data_env
source synthetic_data_env/bin/activate  # (Windows: synthetic_data_env\Scripts\activate)

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Launch Notebook
jupyter notebook

 Project Phases
Phase	Description	Notebook
🟦 Phase 1	Foundation & Data Preparation	01_eda_baseline.ipynb 
🟩 Phase 2	Exploratory Data Analysis	02_comprehensive_eda.ipynb
🟪 Phase 3	Synthetic Data Generation	03_synthetic_data_generation.ipynb
🟧 Phase 4	Model Training & Evaluation	04_model_training_evaluation.ipynb
🟥 Phase 5	Documentation & Deployment	05_documentation_deployment.ipynb
📈 Key Results
📊 Performance Improvements

F1-Score: +15–25% improvement over baseline

Recall: 85%+ fraud detection rate

Precision: Controlled false positives

💼 Business Impact

Net Benefit: Significant cost savings

ROI: Measurable return from improved fraud detection

Risk Reduction: Enhanced protection against financial losses

🔬 Technical Innovations

Statistically validated synthetic data

Comparative analysis of CTGAN, VAE, SMOTE

Scalable and production-ready architecture

🏁 Getting Started

Run Phase 1 to set up and analyze data

Perform EDA in Phase 2

Generate synthetic data in Phase 3

Train and evaluate models in Phase 4

Review documentation & results in Phase 5
