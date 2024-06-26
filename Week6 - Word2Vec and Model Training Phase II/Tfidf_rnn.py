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
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

project_root = "E:\\uvic\\Summer-2024-ECE-597-Group8-main"
random_state = 42
test_size = 0.1

df = pd.read_csv(
    os.path.join(project_root, "data", "processed", "features_tfidf_labels.csv")
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
    y_scores = model.predict(X).ravel()

    # Convert decision scores to binary predictions using 0.5 as a threshold
    y_pred = (y_scores > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_scores)
    average_precision = average_precision_score(y, y_scores)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')

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

    return accuracy, roc_auc, average_precision, precision, recall, f1

def bool_to_int(x):
    return x.astype(int)

def log_transform(x):
    return np.log1p(x)

# Pipelines for different scaling methods
log_transform_pipeline_RobustScaler = Pipeline(
    steps=[
        ('log_transform', FunctionTransformer(log_transform)),
        ('scaler', RobustScaler())
    ]
)

# Column transformer for feature preprocessing
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

# Data splitting
X = df.drop(columns=["Label"])
y = df["Label"]
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Preprocess features
X_processed = features_preprocessor.fit_transform(X_train_val)

# Data reshaping for RNN
X_processed_reshaped = X_processed.reshape((X_processed.shape[0], 1, X_processed.shape[1]))

def create_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# k-fold validation
kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
results = []

for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_processed_reshaped[train_index], X_processed_reshaped[val_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

    # Build and train the RNN model
    rnn_model = create_rnn_model((X_train.shape[1], X_train.shape[2]))
    rnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model on the validation set
    y_val_pred = rnn_model.predict(X_val).ravel()
    y_val_pred_binary = (y_val_pred > 0.5).astype(int)
    results.append(evaluate_model(rnn_model, X_val, y_val))

# Calculate average results
avg_results = np.mean(results, axis=0)
print(f"Validation set evaluation (average over folds):")
print(f"Accuracy: {avg_results[0]:.2f}")
print(f"ROC-AUC Score: {avg_results[1]:.2f}")
print(f"Average Precision-Recall Score: {avg_results[2]:.2f}")
print(f"Precision: {avg_results[3]:.2f}")
print(f"Recall: {avg_results[4]:.2f}")
print(f"F1 Score: {avg_results[5]:.2f}")

# Final evaluation on the test set
X_test_processed = features_preprocessor.transform(X_test)
X_test_processed_reshaped = X_test_processed.reshape((X_test_processed.shape[0], 1, X_test_processed.shape[1]))
rnn_model = create_rnn_model((X_test_processed_reshaped.shape[1], X_test_processed_reshaped.shape[2]))
rnn_model.fit(X_processed_reshaped, y_train_val, epochs=10, batch_size=32, verbose=0)
test_results = evaluate_model(rnn_model, X_test_processed_reshaped, y_test, plot_curves=True)
print(f"Test set evaluation:")
print(f"Accuracy: {test_results[0]:.2f}")
print(f"ROC-AUC Score: {test_results[1]:.2f}")
print(f"Average Precision-Recall Score: {test_results[2]:.2f}")
print(f"Precision: {test_results[3]:.2f}")
print(f"Recall: {test_results[4]:.2f}")
print(f"F1 Score: {test_results[5]:.2f}")
