import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    roc_curve, balanced_accuracy_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

random_state = 42
test_size = 0.1
df = pd.read_csv(
    "/Users/yuecao/Documents/UVic courses/Summer-2024-ECE-597-Group8/data/processed/features_tfidf_labels.csv")
df.info()
df.describe()

def evaluate_model(model, X_train, y_train, X_test, y_test, cv=3):
    # Get decision function scores via cross-validation on the training set
    y_train_probas = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')
    y_train_scores = y_train_probas[:, 1]  # random forest predict positive probability

    # Convert decision scores to binary predictions using 0 as a threshold
    y_train_pred = (y_train_scores > 0.5).astype(int)

    print("Training Set Evaluation:")
    print("========================")
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_train, y_train_pred, zero_division=1))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))

    # Calculate and print accuracy
    accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # balanced accuracy
    balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    print(f"Balanced Accuracy: {balanced_acc:.2f}")

    # Calculate ROC-AUC and Precision-Recall
    roc_auc = roc_auc_score(y_train, y_train_scores)
    average_precision = average_precision_score(y_train, y_train_scores)

    # Print ROC-AUC and Average Precision-Recall scores
    print(f"ROC-AUC Score: {roc_auc:.2f}")
    print(f"Average Precision-Recall Score: {average_precision:.2f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_train, y_train_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Training Set ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_train, y_train_scores)
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (area = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Training Set Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.show()

    # Evaluate on test set
    y_test_probas = model.predict_proba(X_test)
    y_test_scores = y_test_probas[:, 1]
    y_test_pred = model.predict(X_test)

    print("Test Set Evaluation:")
    print("====================")
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=1))

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    print(f"Balanced Accuracy: {balanced_acc:.2f}")

    # Calculate ROC-AUC and Precision-Recall
    roc_auc = roc_auc_score(y_test, y_test_scores)
    average_precision = average_precision_score(y_test, y_test_scores)

    # Print ROC-AUC and Average Precision-Recall scores
    print(f"ROC-AUC Score: {roc_auc:.2f}")
    print(f"Average Precision-Recall Score: {average_precision:.2f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Set ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_scores)
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (area = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Test Set Precision-Recall Curve')
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
    GaussianNB(),
)

model.fit(X_train, y_train)
evaluate_model(model, X_train, y_train, X_test, y_test)

model = make_pipeline(
    features_preprocessor,
    PCA(n_components=0.95),
    GaussianNB(),
)

model.fit(X_train, y_train)
evaluate_model(model, X_train, y_train, X_test, y_test)

model = make_pipeline(
    features_preprocessor,
    LDA(),
    GaussianNB(),
)

model.fit(X_train, y_train)
evaluate_model(model, X_train, y_train, X_test, y_test)
