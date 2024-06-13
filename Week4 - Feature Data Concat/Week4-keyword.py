import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_clean_data(phishing_path, real_path):
    phishing_data = pd.read_csv(phishing_path)
    real_emails = pd.read_csv(real_path)

    phishing_data = phishing_data.dropna(subset=['Body'])
    real_emails = real_emails.dropna(subset=['message'])

    phishing_data['cleaned_text'] = phishing_data['Body'].apply(clean_text)
    real_emails['cleaned_text'] = real_emails['message'].apply(clean_text)

    phishing_data = phishing_data[phishing_data['cleaned_text'].str.strip() != '']
    real_emails = real_emails[real_emails['cleaned_text'].str.strip() != '']

    return phishing_data, real_emails


def clean_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text


def extract_keywords_and_tfidf(phishing_data, real_emails, top_n=200):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    X_phishing_tfidf = tfidf.fit_transform(phishing_data['cleaned_text']).toarray()
    X_real_tfidf = tfidf.transform(real_emails['cleaned_text']).toarray()

    X_combined = np.vstack((X_phishing_tfidf, X_real_tfidf))
    y_combined = np.hstack((np.ones(X_phishing_tfidf.shape[0]), np.zeros(X_real_tfidf.shape[0])))

    feature_names = np.array(tfidf.get_feature_names_out())

    mean_phishing_tfidf = np.mean(X_phishing_tfidf, axis=0)
    mean_real_tfidf = np.mean(X_real_tfidf, axis=0)
    tfidf_difference = mean_phishing_tfidf - mean_real_tfidf

    top_indices = np.argsort(tfidf_difference)[-top_n:]
    significant_keywords = feature_names[top_indices]

    return significant_keywords, X_combined, y_combined


def add_keyword_feature(data, significant_keywords):
    def keyword_match(text):
        words = text.split()
        return sum(1 for word in words if word in significant_keywords)

    data['keyword_matches'] = data['cleaned_text'].apply(keyword_match)
    return data[['keyword_matches']]


# Main function to orchestrate the data processing and feature extraction
def main():
    phishing_data, real_emails = load_and_clean_data('CaptstoneProjectData_2024.csv', 'emails.csv')
    significant_keywords, X_combined, y_combined = extract_keywords_and_tfidf(phishing_data, real_emails)

    # Adding keyword feature to the data
    phishing_keyword_feature = add_keyword_feature(phishing_data, significant_keywords)
    real_keyword_feature = add_keyword_feature(real_emails, significant_keywords)

    # Combined keyword match feature
    X_keywords_combined = np.vstack((phishing_keyword_feature, real_keyword_feature))

    print("Combined keyword match feature matrix shape:", X_keywords_combined.shape)
    print("Labels shape:", y_combined.shape)

    # Print the first few rows of X_combined and y_combined
    print("First few rows of X_combined:\n", X_combined[:5])
    print("First few rows of y_combined:\n", y_combined[:5])

    # # Save significant keywords to CSV
    # keywords_df = pd.DataFrame(significant_keywords, columns=['Keyword'])
    # keywords_df.to_csv('significant_keywords.csv', index=False)
    # print("Significant keywords saved to 'significant_keywords.csv'")
    #

# Call main function
if __name__ == "__main__":
    main()
