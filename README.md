## Project Overview

Diabetic Retinopathy (DR) is a major cause of blindness, affecting ~35% of diabetics globally. Early detection is critical but challenging without image-based diagnostics. This project tackles DR prediction **without images**, using **structured tabular data** and advanced techniques like **SMOTENC**, **synthetic data generation**, and **deep neural networks in PyTorch**.

---

## Objectives

- Preprocess tabular data with class imbalance (~10:1).
- Apply outlier detection (IQR-based) and oversampling (SMOTENC).
- Generate synthetic samples using SDV’s **TVAE** and **CTGAN**.
- Train and evaluate deep learning models with proper validation strategies.
- Optimize hyperparameters using **Optuna**.
- Ensure reproducibility through modular code and documentation.

---

## Results

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | 83.81%    |
| F1 Score     | 38.0%     |
| Precision    | 30.97%    |
| Recall       | 49.14%    |
| ROC AUC      | 0.6842    |

Despite high accuracy, class imbalance affects precision and recall. Further tuning and regularization may improve minority class detection.

---

##  Installation & Setup

```bash
git clone https://github.com/TanLeZhan/ADL2.git
cd ADL2
pip install -r requirements.txt
