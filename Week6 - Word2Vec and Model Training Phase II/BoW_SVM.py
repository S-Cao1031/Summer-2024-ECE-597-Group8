import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

project_root = "E:\\uvic\\Summer-2024-ECE-597-Group8-main"
random_state = 42
test_size = 0.1

df = pd.read_csv(
    os.path.join(project_root, "data", "processed", "features_bow_labels.csv")
)

def evaluate_model(model, X, y, plot_curves=False):
    """
    Evaluates the performance of a binary classifier.

    Args:
    model (estimator): The machine learning model to evaluate.
    X (DataFrame or array-like): Feature data used for training the model.
    y (array-like): True labels.
    plot_curves (bool): Whether to plot ROC and Precision-Recall curves.

    Returns:
    tuple: Accuracy, ROC-AUC score, Average precision-recall score, precision, recall, f1 score
    """
    # Get decision function scores
    y_scores = model.decision_function(X)

    # Convert decision scores to binary predictions using 0 as a threshold
    y_pred = (y_scores > 0).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_scores)
    average_precision = average_precision_score(y, y_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    balanced_accuracy = balanced_accuracy_score(y, y_pred)

    if plot_curves:
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

        # Plot Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y, y_scores)
        plt.figure()
        plt.plot(recall_curve, precision_curve, label=f'Precision-Recall curve (area = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.show()

    return accuracy, roc_auc, average_precision, precision, recall, f1, balanced_accuracy

def bool_to_int(x):
    return x.astype(int)

def log_transform(x):
    return np.log1p(x)

# Pipelines for different scaling methods
homoglyphs_pipeline = make_pipeline(FunctionTransformer(log_transform))
homoglyphs_pipeline_MinMax = make_pipeline(
    FunctionTransformer(log_transform), MinMaxScaler()
)
homoglyphs_pipeline_standardScaler = make_pipeline(
    FunctionTransformer(log_transform), StandardScaler()
)
log_transform_pipeline_RobustScaler = make_pipeline(
    FunctionTransformer(log_transform), RobustScaler()
)

bow_log_transform = FunctionTransformer(np.log1p, validate=True)
bow_transformer_robust = make_pipeline(bow_log_transform,RobustScaler())

bow_transformer_min_max = make_pipeline(
    FunctionTransformer(np.log1p, validate=True),  # Apply logarithmic transformation
    MinMaxScaler()  # Use MinMaxScaler to scale
)

bow_transformer_standard = make_pipeline(
    FunctionTransformer(np.log1p, validate=True),  # Apply logarithmic transformation
    StandardScaler()  # Standardize with StandardScaler
)

features_preprocessor = ColumnTransformer(
    transformers=[
        (
            "log_transform",
            log_transform_pipeline_RobustScaler,
            [
                "Word_Count",
                "Homoglyphs",
                "Total_Abnormal_Count",
                "html_tags",
                "js_code",
            ],
        ),
        (
            "bow_log_transform",
            bow_transformer_standard,
            [col for col in df.columns if col.startswith('BoW_')]
        ),
    ],
    remainder="passthrough",
)

# Data splitting
X = df.drop(columns=["Label"])
y = df["Label"]
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# k-fold validation
kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
results = []

for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

    # Build and train the model
    model = make_pipeline(
        features_preprocessor,
        SVC(
            C=1.0, kernel="rbf", gamma="scale", random_state=random_state
        ),
    )
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    results.append(evaluate_model(model, X_val, y_val))

# Calculate average results
avg_results = np.mean(results, axis=0)
print(f"Validation set evaluation (average over folds):")
print(f"Accuracy: {avg_results[0]:.2f}")
print(f"ROC-AUC Score: {avg_results[1]:.2f}")
print(f"Average Precision-Recall Score: {avg_results[2]:.2f}")
print(f"Precision: {avg_results[3]:.2f}")
print(f"Recall: {avg_results[4]:.2f}")
print(f"F1 Score: {avg_results[5]:.2f}")
print(f"Balanced Accuracy: {avg_results[6]:.2f}")

# Final evaluation on the test set
print("Test set evaluation:")
model.fit(X_train_val, y_train_val)
test_results = evaluate_model(model, X_test, y_test, plot_curves=True)
print(f"Accuracy: {test_results[0]:.2f}")
print(f"ROC-AUC Score: {test_results[1]:.2f}")
print(f"Average Precision-Recall Score: {test_results[2]:.2f}")
print(f"Precision: {test_results[3]:.2f}")
print(f"Recall: {test_results[4]:.2f}")
print(f"F1 Score: {test_results[5]:.2f}")
print(f"Balanced Accuracy: {test_results[6]:.2f}")
