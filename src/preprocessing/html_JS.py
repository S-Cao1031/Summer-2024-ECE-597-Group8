import re
import pandas as pd

def extract_features_from_message(message):
    # Preprocess the email text to extract the subject and clean the body
    subject_match = re.search(r"Subject: (.*)", message)
    subject = subject_match.group(1) if subject_match else "Subject Not Found"
    cleaned_message = re.sub(r"Message-ID:.*?X-FileName:.*?\n", "", message, flags=re.S)

    # Extract number of HTML tags and JavaScript code blocks
    html_tags = len(re.findall(r'<[^>]+>', cleaned_message))
    js_code = len(re.findall(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', cleaned_message))

    return html_tags, js_code

def process_email_text(text):
    # Extract the subject and clean the text
    subject_match = re.search(r"Subject: (.*)", text)
    subject = subject_match.group(1) if subject_match else "Subject Not Found"
    cleaned_text = re.sub(r"Message-ID:.*?X-FileName:.*?\n", "", text, flags=re.S)
    return subject, cleaned_text

def main():
    # Load the datasets
    emails_normal = pd.read_csv('emails.csv', usecols=['message'])
    emails_phishing = pd.read_csv('CaptstoneProjectData_2024.csv')

    # Apply the feature extraction
    emails_normal['features'] = emails_normal['message'].apply(extract_features_from_message)
    emails_phishing['features'] = emails_phishing.apply(
        lambda row: extract_features_from_message(
            str(row['Subject']) + ' ' + str(row['Body'])
        ), axis=1
    )

    # Convert the extracted features to DataFrame
    columns = ['html_tags', 'js_code']
    emails_normal_features_df = pd.DataFrame(emails_normal['features'].tolist(), columns=columns)
    emails_phishing_features_df = pd.DataFrame(emails_phishing['features'].tolist(), columns=columns)

    # Output the DataFrames to console
    print(emails_normal_features_df)
    print(emails_phishing_features_df)

    # Optionally save the features to CSV files
    # emails_normal_features_df.to_csv('emails_normal_features1.csv', index=False)
    # emails_phishing_features_df.to_csv('emails_phishing_features1.csv', index=False)

if __name__ == "__main__":
    main()
