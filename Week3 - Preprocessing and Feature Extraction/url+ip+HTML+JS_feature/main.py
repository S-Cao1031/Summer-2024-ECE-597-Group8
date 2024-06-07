import re
from urllib.parse import urlparse
import ipaddress
import pandas as pd


# Load the datasets
emails_normal = pd.read_csv('emails.csv', usecols=['message'])
emails_phishing = pd.read_csv('CaptstoneProjectData_2024.csv')



def sanitize_url(url):
    # Strip unwanted characters that may corrupt URL parsing:
    # Remove trailing periods, brackets, or commas that often appear in malformed URLs
    cleaned_url = re.sub(r'[\],.]+$','', url)  # This regex now also strips trailing dots which are not typical in hostnames.
    # Remove square brackets from around non-IP address sections of the URL
    cleaned_url = re.sub(r'\[(\D+?)\]', r'\1', cleaned_url)
    return cleaned_url

def extract_advanced_features(message):
    # Extract URLs based on a regex pattern for http/https
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)
    num_urls = len(urls)
    valid_domains = set()

    for url in urls:
        cleaned_url = sanitize_url(url)
        try:
            parsed_url = urlparse(cleaned_url)
            domain = parsed_url.netloc
            # Validate if the domain is an IP address; if valid, add to domains
            try:
                ipaddress.ip_address(domain)
                valid_domains.add(domain)
            except ValueError:
                # Not an IP address; if it's a valid domain and non-empty, add it
                if domain:
                    valid_domains.add(domain)
        except Exception as e:
            print(f"An error occurred while parsing URL: {url}, error: {e}")

    num_domains = len(valid_domains)
    num_ip_addresses = len(re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', message))
    html_tags = bool(re.search(r'<[^>]+>', message))
    js_code = bool(re.search(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', message))
    return num_urls, num_domains, num_ip_addresses, html_tags, js_code


def extract_features_from_message(message):
    # Preprocess the email text to extract the subject and clean the body
    subject, cleaned_message = process_email_text(message)

    # Use the cleaned message for further feature extraction
    length = len(cleaned_message)
    num_links = len(re.findall(r'http[s]?://', cleaned_message))
    num_special_chars = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', cleaned_message))
    keywords = ['urgent', 'verify', 'account', 'password', 'click', 'login', 'update']
    num_keywords = sum(cleaned_message.lower().count(keyword) for keyword in keywords)
    advanced_features = extract_advanced_features(cleaned_message)

    # Optionally, you can include the subject as part of the features if needed
    # Or use it in some other part of your analysis
    return (length, num_links, num_special_chars, num_keywords) + advanced_features


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
columns = ['length', 'num_links', 'num_special_chars', 'num_keywords', 'num_urls', 'num_domains', 'num_ip_addresses', 'html_tags', 'js_code']
emails_normal_features_df = pd.DataFrame(emails_normal['features'].tolist(), columns=columns)
emails_phishing_features_df = pd.DataFrame(emails_phishing['features'].tolist(), columns=columns)

# Save the extracted features to CSV files
emails_normal_features_df.to_csv('emails_normal_features1.csv', index=False)
emails_phishing_features_df.to_csv('emails_phishing_features1.csv', index=False)

print("Normal Emails Features saved to 'emails_normal_features.csv'")
print("Phishing Emails Features saved to 'emails_phishing_features.csv'")