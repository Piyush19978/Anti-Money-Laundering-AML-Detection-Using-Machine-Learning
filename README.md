# Anti-Money-Laundering-AML-Detection-Using-Machine-Learning
Detecting money laundering in the IBM AML HI-Small dataset using Logistic Regression, Random Forest, XGBoost, LightGBM and MLP

## Overview
This project detects money laundering transactions using machine learning.
It compares linear models, tree-based ensembles and a neural network on a
highly imbalanced dataset (about 0.10% laundering transactions).

## Dataset
IBM Transactions for Anti-Money Laundering (AML), HI-Small version.
Source: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

The dataset is not included in this repository because of its size.
Download `HI-Small_Trans.csv` from Kaggle and upload it manually when
Notebook 1 asks for it.

## Project Structure
```
├── README.md
├── requirements.txt
├── report/
│   └── Report.pdf
├── notebooks/
│   ├── Group_Notebook_1_Data_Prep.ipynb
│   ├── Akash_Modelling_CW_F.ipynb
│   ├── Parth_Modelling_Notebook_final.ipynb
│   ├── Piyush_Modelling_Notebook_final.ipynb
│   ├── Nagashri_Modeling.ipynb
│   └── Group_Notebook_2_Visualisation.ipynb
└── data/
    └── cross_group_comparison_final.csv
```

## Methodology
1. **Data preparation (Notebook 1):** loading, EDA, duplicate removal,
   time features, train/test split (80/20, stratified), SMOTE on the
   training set only (minority raised to 10% of majority).
2. **Modelling (individual notebooks):** each member trains a Logistic
   Regression variant plus one other model.
3. **Evaluation and dashboard (Notebook 2):** compares all models using
   Precision, Recall, F1, ROC-AUC and AUPRC.

## Models
| Author | Model 1 | Model 2 |
|---|---|---|
| Akash | Logistic Regression (L1) | Random Forest (tuned) |
| Parth | Logistic Regression (no penalty) | XGBoost |
| Piyush | Logistic Regression (ElasticNet) | LightGBM |
| Nagashri | Logistic Regression (L2) | MLP (128, 64) |

## Results
| Model | Threshold | Precision | Recall | F1 | ROC-AUC | AUPRC |
|---|---|---|---|---|---|---|
| LR (L1) | 0.5 | 0.03 | 0.14 | 0.05 | 0.914 | 0.01 |
| Random Forest (tuned) | 0.5 | 0.16 | 0.31 | 0.21 | 0.913 | 0.17 |
| LR (No Penalty) | 0.5 | 0.027 | 0.177 | 0.047 | 0.894 | 0.029 |
| XGBoost (default) | 0.5 | 0.008 | 0.925 | 0.016 | 0.968 | 0.196 |
| XGBoost (tuned) | 0.93 | 0.015 | 0.804 | 0.030 | 0.968 | 0.196 |
| LR (ElasticNet) | 0.5 | 0.001 | 0.299 | 0.003 | 0.533 | 0.023 |
| LightGBM | 0.5 | 0.029 | 0.675 | 0.055 | 0.968 | 0.241 |
| LR (L2) | 0.857 | 0.056 | 0.053 | 0.054 | 0.899 | 0.022 |
| MLP (128, 64) | 0.494 | 0.034 | 0.179 | 0.057 | 0.887 | 0.027 |

**Key finding:** LightGBM has the best AUPRC (0.241). XGBoost has the
highest recall but very low precision. Accuracy is not used because of
the extreme class imbalance.

## How to Run
1. Open the notebooks in Google Colab.
2. Run `Group_Notebook_1_Data_Prep.ipynb` first. It produces
   `X_train_balanced.parquet`, `X_test_processed.parquet`,
   `y_train_balanced.npy` and `y_test.npy`.
3. Run any modelling notebook, uploading those four files when asked.
4. Run `Group_Notebook_2_Visualisation.ipynb` last. Update the CSV
