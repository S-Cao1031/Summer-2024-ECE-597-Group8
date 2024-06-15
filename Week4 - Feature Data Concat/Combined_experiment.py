import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


#prepare dataset for experiment
#for PCA analysis
def apply_pca(xTrain, xTest, n_components):
    pca = PCA(n_components=n_components)
    pca_train_features = pca.fit_transform(xTrain)
    pca_test_features = pca.transform(xTest)
    return pca_train_features, pca_test_features


#for FLD analysis
def apply_lda(xTrain, xTest, labels, n_components):
    lda = LDA(n_components=n_components)
    lda_train_features = lda.fit_transform(xTrain, labels)
    lda_test_features = lda.transform(xTest)
    return lda_train_features, lda_test_features


def show_result(model, x_train, x_test, y_train, y_test):
    # make prediction
    y_pred = model.predict(x_test)

    # print result
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # print confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # print ROC
    y_scores = cross_val_predict(model, x_train, y_train, cv=3, method="decision_function")
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


###start from here####
df = pd.read_csv('features_tfidf_labels.csv')
features = df.iloc[:, : - 1]
labels = df.iloc[:, -1]

#modify the test size here, random size is fixed for now
test_size = 0.9
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
findGlyph = x_train[x_train['Homoglyphs'] > 1]
print(findGlyph)
#preprocess data here
columns_to_scale = ['Homoglyphs']
other_columns = [col for col in features.columns if col not in columns_to_scale]

preprocessorM = ColumnTransformer(
    transformers=[
        ('minmax', MinMaxScaler(), columns_to_scale),
        ('passthrough', 'passthrough', other_columns)
    ]
)
preprocessorS = ColumnTransformer(
    transformers=[
        ('standard', StandardScaler(), columns_to_scale),
        ('passthrough', 'passthrough', other_columns)
    ]
)

x_train_scaled = preprocessorM.fit_transform(x_train)
x_test_scaled = preprocessorM.transform(x_test)
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=columns_to_scale + other_columns)
x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=columns_to_scale + other_columns)
homoglyphs_train_scaled = x_train_scaled_df['Homoglyphs']
find_scaled_glyph = homoglyphs_train_scaled > 0
print("Standardized 'Homoglyphs' Column in Training Data:")
print(x_train_scaled_df[find_scaled_glyph])


#pcaTrain, pcaTest = apply_pca(x_train_scaled_df, x_test_scaled_df, 10)
pcaTrain, pcaTest = apply_pca(x_train, x_test, 10)
#ldaTrain, ldaTest = apply_lda(x_train_scaled_df, x_test_scaled_df, y_train, 1)
ldaTrain, ldaTest = apply_lda(x_train,x_test,y_train,1)
#Use SVM first; as a standard testing
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
#svm_model.fit(pcaTrain,y_train)

#show_result(svm_model,pcaTrain,pcaTest,y_train,y_test)

svm_model.fit(ldaTrain, y_train)
show_result(svm_model, ldaTrain, ldaTest, y_train, y_test)
