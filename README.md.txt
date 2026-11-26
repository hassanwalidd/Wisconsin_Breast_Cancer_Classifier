# Early Detection of Breast Cancer

I built this to predict whether a breast tumor is malignant or benign from measurements of cell nuclei. I used it to practice a full classification workflow on a small dataset, and to make sure I avoided data leakage.

## Data

The Wisconsin Breast Cancer (Diagnostic) dataset: 569 records, 30 numeric features, and a diagnosis column (357 benign, 212 malignant). It's available on Kaggle and the UCI repository. Put it in the project folder as `data.csv`.

## What I did

1. Dropped the `id` column and the empty `Unnamed: 32` column, and checked for missing values and duplicates (none found).
2. Encoded the diagnosis (B = 0, M = 1).
3. Split the data 80/20 with a stratified split (`random_state=42`).
4. Fit `StandardScaler` on the training set only, then applied it to the test set.
5. Applied SMOTE to the training set only, so the test set stays untouched.
6. Trained four models: Logistic Regression, SVM (RBF kernel), Random Forest and Decision Tree (both with `max_depth=15`).
7. Evaluated each model on the test set with accuracy, precision, recall, F1 and ROC-AUC. AUC is computed from predicted probabilities, not hard labels.

## Results (test set, 114 samples)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 97.37 | 97.56 | 95.24 | 96.39 | 99.27 |
| SVM | 97.37 | 97.56 | 95.24 | 96.39 | 99.57 |
| Random Forest | 96.49 | 97.50 | 92.86 | 95.12 | 99.74 |
| Decision Tree | 91.23 | 86.36 | 90.48 | 88.37 | 91.07 |

Logistic Regression and SVM did best on accuracy and F1. The Decision Tree was clearly the weakest.

## Limitations

- The test set is small. One wrong prediction changes accuracy by about 0.9 points, so I wouldn't read much into small gaps between models.
- This dataset is clean and easy to separate, so high scores are expected. It doesn't tell me how the models would do on messier clinical data.
- I used one train/test split and did not tune hyperparameters.
- SMOTE may not be needed here, since the classes are only mildly imbalanced (about 63% / 37%).

## Run it

```
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook
```

Open the notebook and run all cells.
