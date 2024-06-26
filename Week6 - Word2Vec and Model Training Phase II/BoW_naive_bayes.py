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
    "/Users/yuecao/Documents/UVic courses/Summer-2024-ECE-597-Group8/data/processed/features_bow_labels.csv")
df.info()
df.describe()

def evaluate_model(model, X, y, cv=3):
    # Get decision function scores via cross-validation
    y_probas = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
    y_scores = y_probas[:, 1] # randome forest predict positive probablitity

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

    # balanced accuracy
    balanced_acc = balanced_accuracy_score(y, y_pred)
    print(f"Balanced Accuracy: {balanced_acc:.2f}")

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

bow_log_transform = FunctionTransformer(np.log1p, validate=True)
bow_transformer_robust = make_pipeline(bow_log_transform,RobustScaler())

bow_transformer_min_max = make_pipeline(
    FunctionTransformer(np.log1p, validate=True),  # 应用对数变换
    MinMaxScaler()  # 使用 MinMaxScaler 进行缩放
)

bow_transformer_standard = make_pipeline(
    FunctionTransformer(np.log1p, validate=True),  # 应用对数变换
    StandardScaler()  # 使用 StandardScaler 进行标准化
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
model = make_pipeline(
    features_preprocessor,
    GaussianNB(),
)
model.fit(X_train, y_train)
evaluate_model(model, X_train, y_train)

'''
model = make_pipeline(
    features_preprocessor,
    PCA(n_components=0.95),
    GaussianNB(),
)

model.fit(X_train, y_train)
evaluate_model(model, X_train, y_train)
# 朴素贝叶斯分类器基于特征间的条件独立性假设，而PCA通过线性组合生成新的特征，这些新特征之间可能不满足条件独立性。因此，应用PCA后直接使用朴素贝叶斯可能会影响模型的性能。
'''

model = make_pipeline(
    features_preprocessor,
    LDA(),
    GaussianNB(),
)

model.fit(X_train, y_train)
evaluate_model(model, X_train, y_train)