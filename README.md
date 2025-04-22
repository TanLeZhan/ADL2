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

## Dataset

A population-based [dataset](https://doi.org/10.1371/journal.pone.0275617.s001) of 6,374 diabetic patients from a Chinese cohort, including clinical biomarkers and demographic data.

**Reference**: [Predicting the risk of diabetic retinopathy using explainable machine learning algorithms - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1871402123002151#sec2.2.2)

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
```
After downloading the dataset, place it in the ```original_dataset``` folder and rename it as ```raw_data```. Run the following command below to get a cleaned dataset labelled ```preprocessed_data_OHE```.
```bash
python data_processing.py
```

To get the dataset labelled as ```preprocessed_data_encoded```, run ```Input Dataset.ipynb```

Run the notebook ```Final Dataset Generator.ipynb``` to get the final augmented input data set for model training.

Run the notebook ```Final Pipeline for report.ipynb```  to train the model and get model performance results.

