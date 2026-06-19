# DES Viscosity Prediction with Machine Learning


This repository contains the code and data for the manuscript:

**"Machine Learning with System-Disjoint Nested Cross-Validation for DES Viscosity Prediction: GFN2-xTB Descriptors and SHAP Interpretability"**

## 📋 Overview

- **Purpose**: Predict viscosity of Deep Eutectic Solvents (DES) using machine learning
- **Key Feature**: System-disjoint nested cross-validation (no DES system overlap between train/test)
- **Models**: Random Forest (best), Gradient Boosting, XGBoost, and 6 other baselines
- **Interpretability**: SHAP analysis (TreeExplainer) with feature importance and dependence plots
- **Descriptors**: GFN2-xTB quantum-chemical descriptors + composition/temperature features

## 📊 Results Summary

| Model | R² (nested CV) | RMSLE | RMSE (cP) |
|-------|---------------|-------|-----------|
| Random Forest | 0.658 ± 0.036 | 0.145 ± 0.019 | 115.8 ± 18.2 |
| Gradient Boosting | 0.637 ± 0.041 | 0.152 ± 0.021 | 124.3 ± 22.1 |
| XGBoost | 0.622 ± 0.048 | 0.158 ± 0.026 | 131.7 ± 27.4 |
| Arrhenius Baseline | 0.441 ± 0.082 | 0.216 ± 0.034 | — |

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/des-viscosity-ml-prediction.git
cd des-viscosity-ml-prediction

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
