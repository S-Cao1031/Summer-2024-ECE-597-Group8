import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mstats
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve, 
    roc_auc_score, 
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC

random_state = 42
test_size = 0.1

df = pd.read_csv(
    "/Users/yuecao/Documents/UVic courses/Summer-2024-ECE-597-Group8/data/processed/features_bow_labels.csv")

df.info()

df.describe()

df.hist()
plt.show()

features = ['Word_Count', 'Homoglyphs', 'Total_Abnormal_Count', 'html_tags']

# Setup the plot
fig, axes = plt.subplots(nrows=len(features), ncols=2, figsize=(12, 5 * len(features)))

# Loop through each feature
for i, feature in enumerate(features):
    # Apply log transformation
    log_transformed = np.log1p(df[feature])
    
    # Plot original distribution
    sns.histplot(df[feature], kde=True, ax=axes[i, 0], color='blue')
    axes[i, 0].set_title(f'Original Distribution of {feature}')
    axes[i, 0].set_xlabel(f'{feature}')
    axes[i, 0].set_ylabel('Frequency')
    
    # Plot log transformed distribution
    sns.histplot(log_transformed, kde=True, ax=axes[i, 1], color='green')
    axes[i, 1].set_title(f'Log Transformed Distribution of {feature}')
    axes[i, 1].set_xlabel(f'Log_{feature}')
    axes[i, 1].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

def evaluate_model(model, X, y, cv=3):
    """
    Evaluates the performance of a binary classifier using cross-validation.

    Args:
    model (estimator): The machine learning model to evaluate.
    X (DataFrame or array-like): Feature data used for training the model.
    y (array-like): True labels.
    cv (int): Number of cross-validation folds.

    Returns:
    None
    """
    # Get decision function scores via cross-validation
    y_scores = cross_val_predict(model, X, y, cv=cv, method='decision_function')

    # Convert decision scores to binary predictions using 0 as a threshold
    y_pred = (y_scores > 0).astype(int)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y, y_pred))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Calculate and print accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate ROC-AUC and Precision-Recall
    roc_auc = roc_auc_score(y, y_scores)
    average_precision = average_precision_score(y, y_scores)

    # Print ROC-AUC and Average Precision-Recall scores
    print(f"ROC-AUC Score: {roc_auc:.2f}")
    print(f"Average Precision-Recall Score: {average_precision:.2f}")

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
    precision, recall, _ = precision_recall_curve(y, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (area = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.show()

X = df.drop(columns=["Label"])
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

def bool_to_int(x):
    return x.astype(int)


def log_transform(x):
    return np.log1p(x)


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
    ],
    remainder="passthrough",
)

model = make_pipeline(
    features_preprocessor,
    # PCA(n_components=0.95),
    SVC(
        C=1.0, kernel="rbf", gamma="scale", random_state=random_state
    ),
)

model.fit(X_train, y_train)
evaluate_model(model, X_train, y_train)

# try again with FLD
model = make_pipeline(
    features_preprocessor,
    LDA(),
    SVC(
        C=1.0, kernel="rbf", gamma="scale", random_state=random_state
    ),
)

model.fit(X_train, y_train)
evaluate_model(model, X, y)

