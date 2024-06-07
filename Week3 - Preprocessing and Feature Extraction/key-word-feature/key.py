import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


phishing_data = pd.read_csv('CaptstoneProjectData_2024.csv')
real_emails = pd.read_csv('emails.csv')


phishing_data = phishing_data.dropna(subset=['Body'])
real_emails = real_emails.dropna(subset=['message'])


print(phishing_data['Body'].head())


def clean_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)  # 去除HTML标签
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 去除特殊字符
    return text


phishing_data['cleaned_text'] = phishing_data['Body'].apply(clean_text)
real_emails['cleaned_text'] = real_emails['message'].apply(clean_text)



print(phishing_data['cleaned_text'].head())


print(real_emails['cleaned_text'].head())


phishing_data = phishing_data[phishing_data['cleaned_text'].str.strip() != '']
real_emails = real_emails[real_emails['cleaned_text'].str.strip() != '']


tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_phishing_tfidf = tfidf.fit_transform(phishing_data['cleaned_text']).toarray()
X_real_tfidf = tfidf.transform(real_emails['cleaned_text']).toarray()


X_combined = np.vstack((X_phishing_tfidf, X_real_tfidf))
y_combined = np.hstack((np.ones(X_phishing_tfidf.shape[0]), np.zeros(X_real_tfidf.shape[0])))


feature_names = np.array(tfidf.get_feature_names_out())


mean_phishing_tfidf = np.mean(X_phishing_tfidf, axis=0)
mean_real_tfidf = np.mean(X_real_tfidf, axis=0)
tfidf_difference = mean_phishing_tfidf - mean_real_tfidf


top_n = 200
top_indices = np.argsort(tfidf_difference)[-top_n:]
significant_keywords = feature_names[top_indices]


print("Significant Keywords for Phishing Detection:")
print(significant_keywords)


keywords_df = pd.DataFrame(significant_keywords, columns=['Keyword'])
keywords_df.to_csv('significant_keywords.csv', index=False)

print("Significant keywords saved to 'significant_keywords.csv'")
