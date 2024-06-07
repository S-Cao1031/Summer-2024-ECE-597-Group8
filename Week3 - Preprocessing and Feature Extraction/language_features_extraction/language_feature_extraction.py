import re
import pandas as pd
from langdetect import detect_langs, LangDetectException
from langdetect.detector_factory import DetectorFactory

DetectorFactory.seed = 0  # Set random seeds for repeatability of results

# Define a global counter variable
call_counter = 0

# Load the datasets
emails_normal = pd.read_csv('emails_part_2.csv')  # Read regular mail data set
emails_phishing = pd.read_csv('CaptstoneProjectData_2024.csv')

# Define a function to map language codes to numerical labels
language_map = {
    'en': 1,  # English
    'es': 2,  # Spanish
    'fr': 3,  # French
    'de': 4,  # German
    'zh-cn': 5,  # Simplified Chinese
    'zh-tw': 6,  # Traditional Chinese
    'ru': 7,  # Russian
    'ja': 8,  # Japanese
    'ko': 9,  # Korean
    'it': 10,  # Italian
    'pt': 11,  # Portuguese
    'nl': 12,  # Dutch
    'ar': 13,  # Arabic
    'hi': 14,  # Hindi
    'sv': 15,  # Swedish
    'fi': 16,  # Finnish
    'da': 17,  # Danish
    'no': 18,  # Norwegian
    'pl': 19,  # Polish
    'tr': 20,  # Turkish
    'cs': 21,  # Czech
    'el': 22,  # Greek
    'he': 23,  # Hebrew
    'th': 24,  # Thai
    'uk': 25,  # Ukrainian
    'id': 26,  # Indonesian
    'hu': 27,  # Hungarian
    'bg': 28,  # Bulgarian
    'ro': 29,  # Romanian
    'vi': 30  # Vietnamese
}


# Define a function to extract meaningful character count
def count_meaningful_characters(text):
    meaningful_text = re.sub(r'http[s]?://\S+|[^A-Za-z0-9\s]', '', text)
    return len(meaningful_text.replace(" ", ""))


# Define a function to extract basic features from an email message
def extract_features_from_message(message):
    global call_counter  # Access the global counter variable
    call_counter += 1  # Increment the counter
    print(call_counter)  # Print the counter value

    length = len(message)
    num_links = len(re.findall(r'http[s]?://', message))
    num_special_chars = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', message))
    keywords = ['urgent', 'verify', 'account', 'password', 'click', 'login', 'update']
    num_keywords = sum(message.lower().count(keyword) for keyword in keywords)

    # Detect languages and calculate word counts
    try:
        detected_langs = detect_langs(message)
        language_counts = {lang.lang: 0 for lang in detected_langs}

        words = message.split()
        for word in words:
            try:
                detected_word_lang = detect_langs(word)[0].lang
                if detected_word_lang in language_counts:
                    language_counts[detected_word_lang] += 1
            except LangDetectException:
                pass

        # removes languages with a word count of 0
        language_counts = {lang: count for lang, count in language_counts.items() if count > 0}

        total_meaningful_words = sum(language_counts.values())
        if total_meaningful_words > 0:
            primary_language = max(language_counts, key=language_counts.get)
            primary_language_words = language_counts[primary_language]
            sorted_langs = sorted(language_counts.items(), key=lambda item: item[1], reverse=True)
            second_language = sorted_langs[1][0] if len(sorted_langs) > 1 else 'unknown'
            second_language_words = sorted_langs[1][1] if len(sorted_langs) > 1 else 0
        else:
            primary_language = 'unknown'
            primary_language_words = 0
            second_language = 'unknown'
            second_language_words = 0
        num_languages = len(language_counts)

        # Calculate meaningful word percentage
        meaningful_words_percentage = (total_meaningful_words / len(words)) * 100 if len(words) > 0 else 0
    except LangDetectException:
        primary_language = 'unknown'
        primary_language_words = 0
        second_language = 'unknown'
        second_language_words = 0
        num_languages = 0
        meaningful_words_percentage = 0

    # Map the primary and second languages to their numerical labels
    primary_language_label = language_map.get(primary_language, 0)
    second_language_label = language_map.get(second_language, 0)

    return (length, num_links, num_special_chars, meaningful_words_percentage, num_keywords,
            num_languages, primary_language_label, primary_language_words, second_language_label, second_language_words)


# Define a function to extract subject based on the provided condition
def extract_subject_from_message(message):
    lines = message.split('\n')  # Split the message into a list by row
    subject = ''  # Initializes the theme string
    start_index = -1  # Initialize the start index to -1
    for i, line in enumerate(lines):
        if 'Subject:' in line:  # If the current line contains 'Subject:', indicates the start of the subject
            start_index = i  # Record the row index at the beginning of the topic
            break
    if start_index != -1:  # If the opening line of the topic is found
        for i in range(start_index, len(lines)):  # Walk through the line starting with the topic
            if 'Content-Type:' in lines[i]:  # If the current line contains 'Content-Type:', indicates the end of the topic
                break
            subject += lines[i].replace('Subject:', '').strip() + ' '  # Adds the current line to the subject string
    return subject.strip()  # Returns a topic string with whitespace removed

# Define a function to extract body based on the provided condition
def extract_body_from_message(message):
    lines = message.split('\n')
    body = ''
    start_index = -1
    for i, line in enumerate(lines):
        if 'X-FileName:' in line:
            start_index = i
            break
    if start_index != -1:
        try:
            for i in range(start_index + 2, len(lines)):
                if lines[i].strip() == '':
                    break
                body += lines[i].strip() + ' '
        except:
            body = ''
    return body.strip()

# Split the normal emails into subject and body based on the new logic
emails_normal['Subject'] = emails_normal['message'].apply(extract_subject_from_message)  # Extract the subject
print(emails_normal['Subject'] )
emails_normal['Body'] = emails_normal['message'].apply(extract_body_from_message)  # Extract the body
print(emails_normal['Body'])

# Combine both datasets into one DataFrame
emails_normal['label'] = 0
emails_phishing['label'] = 1
# Combine both datasets into one DataFrame
combined_emails = pd.concat([emails_normal[['Subject', 'Body', 'label']], emails_phishing[['Subject', 'Body', 'label']]],
                            ignore_index=True)

# Fill missing values with empty strings
combined_emails.fillna('', inplace=True)

# Extract features from the combined dataset and create DataFrame including labels
subject_features_df = pd.DataFrame(
    combined_emails['Subject'].apply(lambda x: extract_features_from_message(x)).tolist(),
    columns=['length_subject', 'num_links_subject', 'num_special_chars_subject', 'meaningful_chars_percentage_subject',
             'num_keywords_subject', 'num_languages_subject', 'primary_language_label_subject', 'primary_language_words_subject',
             'second_language_label_subject', 'second_language_words_subject']
)

body_features_df = pd.DataFrame(
    combined_emails['Body'].apply(lambda x: extract_features_from_message(x)).tolist(),
    columns=['length_body', 'num_links_body', 'num_special_chars_body', 'meaningful_chars_percentage_body',
             'num_keywords_body', 'num_languages_body', 'primary_language_label_body', 'primary_language_words_body',
             'second_language_label_body', 'second_language_words_body']
)

# Add the original subject, body, and label columns to the DataFrame
combined_features_df = pd.concat([subject_features_df, body_features_df, combined_emails[['label']]], axis=1)


# Add the original subject, body, and label columns to the DataFrame
combined_features_df['Subject'] = combined_emails['Subject']
combined_features_df['Body'] = combined_emails['Body']
combined_features_df['label'] = combined_emails['label']

# Save the combined dataset to a new CSV file
combined_features_df.to_csv('combined_features_with_language.csv', index=False)
