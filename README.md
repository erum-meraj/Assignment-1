# SVM Classification on Breast Cancer Dataset

Name: Erum Meraj
ROll Number: 2201CS24
Course: CS502 - APR

This project implements **Support Vector Machine (SVM)** classification on the **Breast Cancer Wisconsin dataset** using Python and Scikit-learn.
It demonstrates the full machine learning workflow — from preprocessing to training, evaluation, and visualization.

## Project Overview

The goal of this assignment is to:

- Load and explore the breast cancer dataset.
- Preprocess features using **StandardScaler**.
- Train an **SVM classifier (RBF kernel)** for binary classification.
- Evaluate the model using:

  - Confusion Matrix
  - Classification Report
  - Accuracy Score
  - Cross-validation

- Visualize the dataset using **PCA (2D projection)**.

## Steps in the Implementation

1. **Load Dataset**

   - Uses `sklearn.datasets.load_breast_cancer()`
   - Features include cell nuclei measurements.
   - Targets:

     - `0`: Malignant
     - `1`: Benign

2. **Train-Test Split**

   - 70% training, 30% testing
   - Stratified to maintain class distribution

3. **Feature Scaling**

   - Standardization via `StandardScaler`

4. **Model Training**

   - **SVM with RBF kernel** (`C=2.0`, `gamma="scale"`)

5. **Model Evaluation**

   - Confusion Matrix
   - Precision, Recall, F1-score
   - Accuracy
   - 5-fold Cross-validation

6. **Visualization**

   - PCA reduces high-dimensional data to **2D**
   - Scatter plot shows class separation

## Example Outputs

- **Confusion Matrix**

  ```
  [[62  1]
   [ 2 106]]
  ```

- **Accuracy**

  ```
  Accuracy: 0.97
  Mean CV Accuracy: 0.96
  ```

- **Visualization**
  A 2D PCA plot showing the dataset distribution.

## Requirements

- Python 3.8+
- Libraries:

  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

Install dependencies via:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

1. Clone this repository or copy the script.
2. Install the required dependencies.
3. Run the script:

   ```bash
   python main.py
   ```

4. View results in the terminal and PCA visualization plot.
