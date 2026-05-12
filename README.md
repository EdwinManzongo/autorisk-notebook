# Hybrid CANN + XGBoost Framework for Dynamic Auto-Insurance Premium Optimisation

**MTech Software Engineering — HIT800 Research Project**

A two-phase hybrid actuarial machine-learning framework that combines a **Combined Actuarial Neural Network (CANN)** with **XGBoost** for binary claim-occurrence prediction, followed by SHAP-based explainability and a risk-adjusted premium pricing simulation.

---

## Overview

Traditional actuarial models (GLMs) lack the representational power to capture non-linear feature interactions, while pure deep-learning models sacrifice interpretability. This framework bridges the gap:

1. **Phase 1 — CANN**: A neural network with a GLM skip connection (Wüthrich & Merz, 2019), warm-started from a Logistic Regression. Architecture: 128 → 64 → 32 hidden units + direct GLM logit path.
2. **Phase 2 — Hybrid Boosting**: Raw features are augmented with 32-dimensional CANN embeddings and fed into a tuned XGBoost classifier.
3. **Pricing Engine**: Calibrated claim probabilities are translated into risk-adjusted premiums using a frequency × severity relativity model.

---

## Architecture

```
Input Features
    │
    ├──► GLM (Logistic skip connection, warm-started)
    │
    └──► Deep Network (128 → 64 → 32, BN + Dropout)
              │
              └──► Combined CANN output (P̂_claim)
                        │
                        ▼
         [Raw Features ∥ 32-dim CANN Embeddings]
                        │
                        ▼
              XGBoost (Tuned, SMOTE)
                        │
                        ▼
         Calibrated P(claim) → Premium Multiplier
```

---

## Models Benchmarked

| # | Model | Category |
|---|-------|----------|
| 1 | Logistic Regression | Baseline |
| 2 | Random Forest | Baseline |
| 3 | XGBoost | Baseline |
| 4 | LightGBM | Baseline |
| 5 | Random Forest + SMOTE | SMOTE |
| 6 | XGBoost + SMOTE | SMOTE |
| 7 | XGBoost (Tuned) | Tuned |
| 8 | MLP + XGBoost | Hybrid |
| 9 | CANN (Standalone) | Neural |
| 10 ★ | **Hybrid CANN + XGBoost (Tuned)** | **Proposed** |
| 11 | CANN + LightGBM | Hybrid |
| 12 | Pricing-Focused Calibrated CANN + XGBoost | Pricing |

---

## Evaluation Metrics

- AUC-ROC, F1-Score, Precision, Recall
- Matthews Correlation Coefficient (MCC)
- Average Precision (PR-AUC)
- Log Loss (primary metric for pricing calibration)
- Expected Calibration Error (ECE)
- Brier Score

---

## Premium Pricing Formula

```
Premium = Base Premium × Risk Factor

Risk Factor = 0.5 + 4 · P̂(claim)   →   multiplier range: [0.5×, 4.5×]
```

The pricing model uses **calibrated** probabilities (via `CalibratedClassifierCV`) to ensure actuarially sound expected-loss estimates.

---

## Requirements

```txt
torch
xgboost
lightgbm
shap
imbalanced-learn
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

Install all dependencies:

```bash
pip install torch xgboost lightgbm shap imbalanced-learn scikit-learn pandas numpy matplotlib seaborn joblib
```

> The notebook is designed to run on **Google Colab** and uses `google.colab.files` for dataset upload. A GPU runtime is recommended for CANN training.

---

## Dataset

The notebook expects `Insurance_claims_data.csv` uploaded at runtime (Google Colab file picker). The dataset must contain:

- A binary target column: `claim_status` (0 = no claim, 1 = claim)
- Vehicle features: `max_torque`, `max_power`, engine specs, vehicle age, etc.
- Policy/driver features: age, region, policy tenure, etc.

A stratified subsample of up to **15,000 rows** is used for tractable training.

---

## Outputs

| File | Description |
|------|-------------|
| `cann_model.pt` | Trained CANN PyTorch state dict |
| `xgb_pricing_proposed.json` | Tuned XGBoost model (JSON format) |
| `scaler.joblib` | Fitted `StandardScaler` for inference |
| `calibrated_model.joblib` | Calibrated wrapper (if pricing model used) |
| `01_*.png` … `07_*.png` | EDA and model evaluation plots |
| `05_shap_bar.png` | SHAP feature importance bar chart |
| `05_shap_beeswarm.png` | SHAP beeswarm summary plot |

---

## Usage

1. Open the notebook in Google Colab.
2. Run **Installing & Importing Libraries** to install dependencies.
3. Upload `Insurance_claims_data.csv` when prompted.
4. Execute all cells sequentially.
5. Review the model leaderboard and SHAP plots.
6. The premium pricing simulation runs on the held-out test set automatically.

---

## Key Results

The proposed **Hybrid CANN + XGBoost** model achieves the best balance of:
- Discriminative power (AUC-ROC, MCC)
- Pricing calibration (Log Loss, ECE, Brier Score)
- Interpretability (SHAP values on both raw and CANN-learned features)

---

## References

- Wüthrich, M. V., & Merz, M. (2019). *Yes, we CANN!* ASTIN Bulletin, 49(1), 1–3.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.

---

## Author

**Edwin Manzongo** — MTech Software Engineering, HIT800 Research Project
