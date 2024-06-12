import re
import pandas as pd

# Load the datasets
emails_normal = pd.read_csv('emails.csv', usecols=['message'])
emails_phishing = pd.read_csv('CaptstoneProjectData_2024.csv')

def extract_advanced_features(message):
    # Continue extracting these non-URL related features
    num_ip_addresses = len(re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', message))
    html_tags = bool(re.search(r'<[^>]+>', message))
    js_code = bool(re.search(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', message))
    return num_ip_addresses, html_tags, js_code

def extract_features_from_message(message):
    # Preprocess the email text to extract the subject and clean the body
    subject, cleaned_message = process_email_text(message)

    # Use the cleaned message for further feature extraction
    length = len(cleaned_message)
    num_links = len(re.findall(r'http[s]?://', cleaned_message))
    num_special_chars = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', cleaned_message))
    keywords = ['urgent', 'verify', 'account', 'password', 'click', 'login', 'update']
    num_keywords = sum(cleaned_message.lower().count(keyword) for keyword in keywords)
    num_ip_addresses, html_tags, js_code = extract_advanced_features(cleaned_message)

    return (length, num_links, num_special_chars, num_keywords, num_ip_addresses, html_tags, js_code)

def process_email_text(text):
    # Extract the subject using a regular expression
    subject_match = re.search(r"Subject: (.*)", text)
    subject = subject_match.group(1) if subject_match else "Subject Not Found"

    # Clean the text by removing everything from "Message-ID:" up to "X-FileName:"
    cleaned_text = re.sub(r"Message-ID:.*?X-FileName:.*?\n", "", text, flags=re.S)

    return subject, cleaned_text

# Example application of new feature extraction
emails_normal['features'] = emails_normal['message'].apply(extract_features_from_message)
emails_phishing['features'] = emails_phishing.apply(
    lambda row: extract_features_from_message(
        str(row['Subject']) + ' ' + str(row['Body'])
    ), axis=1
)

# Convert the extracted features to DataFrame
columns = ['length', 'num_links', 'num_special_chars', 'num_keywords', 'num_ip_addresses', 'html_tags', 'js_code']
emails_normal_features_df = pd.DataFrame(emails_normal['features'].tolist(), columns=columns)
emails_phishing_features_df = pd.DataFrame(emails_phishing['features'].tolist(), columns=columns)

# Save the extracted features to CSV files
emails_normal_features_df.to_csv('emails_normal_features1.csv', index=False)
emails_phishing_features_df.to_csv('emails_phishing_features1.csv', index=False)

print("Normal Emails Features saved to 'emails_normal_features.csv'")
print("Phishing Emails Features saved to 'emails_phishing_features.csv'")
