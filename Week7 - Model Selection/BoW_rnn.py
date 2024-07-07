import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    balanced_accuracy_score, f1_score, make_scorer, classification_report
)
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping

##project_root = "E:\\uvic\\Summer-2024-ECE-597-Group8-main"
random_state = 42
test_size = 0.1

#df = pd.read_csv(
#    os.path.join(project_root, "data", "processed", "features_bow_labels.csv")
#)
df = pd.read_csv("features_bow_labels.csv")
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

# Preprocess features
X_processed = features_preprocessor.fit_transform(X_train_val)

# Data reshaping for RNN
X_processed_reshaped = X_processed.reshape((X_processed.shape[0], 1, X_processed.shape[1]))


def create_rnn_model(trial, input_shape):
    # if we don't include any lstm, the performance will be downgraded
    hidden_size = trial.suggest_int('hidden_size', 20, 200)
    num_layers = trial.suggest_int('num_layers', 1, 3)

    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in range(num_layers):
        return_sequences = True if i < num_layers - 1 else False
        model.add(LSTM(hidden_size, return_sequences=return_sequences))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = []

    for train_index, valid_index in skf.split(X_train_val, y_train_val):
        X_train, X_val = X_processed_reshaped[train_index], X_processed_reshaped[valid_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[valid_index]

        model = create_rnn_model(trial,(X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(X_train,y_train,
                  epochs=trial.suggest_int('epochs', 10, 50),
                  batch_size=trial.suggest_int('batch_size', 32, 256),
                  validation_data=(X_val, y_val),
                  callbacks=[early_stopping],
                  verbose=0)

        y_pred = (model.predict(X_val) > 0.5).astype("int32")
        f1 = f1_score(y_val, y_pred, pos_label=1)
        results.append(f1)

    return np.mean(results)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=6)

# Print the best parameters
print('Best trial:')
trial = study.best_trial
print(f'Value: {trial.value}')
print('Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

def best_rnn_model(input_shape, hidden_size, num_layers):
    model = Sequential()
    model.add(Input(shape=input_shape))


    for i in range(num_layers - 1):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=False))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

best_param = study.best_trial.params


model = best_rnn_model((X_processed_reshaped.shape[1], X_processed_reshaped.shape[2]), best_param['hidden_size'],
                       best_param['num_layers'])


history = model.fit(X_processed_reshaped, y_train_val, epochs=best_param['epochs'], batch_size=best_param['batch_size'], validation_split=0.2)


X_test_processed = features_preprocessor.transform(X_test)
X_test_processed_reshaped = X_test_processed.reshape((X_test_processed.shape[0], 1, X_test_processed.shape[1]))
y_pred_probs = model.predict(X_test_processed_reshaped)
y_pred = (y_pred_probs > 0.5).astype(int)

# Generate the classification report
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n")
print(report)