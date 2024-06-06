import pandas as pd
import re

#   read CSV, it now has 2 columns, Subject & Body
data = 'CaptstoneProjectData_2024.csv'
phishing = pd.read_csv(data)
#   We DROP any row that miss Subject OR Body
phishing = phishing.dropna(subset=['Subject', 'Body'])
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


#   print(phishing)
#   input is the data,output contains 3 features
def extract_features(data):

    #define patterns
    all_patterns = {
        'Amount': r'\$\s?\d+\.?\d*',  # CAD dollar, if it's the case CAD 250.00, that should be the case of keyword
        # r'\b[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b',#can't find
        # r'\b1Z[A-Z0-9]{16}\b', #ups tracking
        # r'\b(?!\+\d{11,14}\b)\d{12,15}\b' # fex tracking
        'Domestic_Phone': r'\b(\+1[-\s]?)?\d{3}[-\s]\d{3}[-\s]\d{4}\b', # +1 | 1 | nothing + xxx xxx xxxx or xxx-xxx-xxxx
        'International_Phone': r'\+(?!1)\d{1,3}\s?\d{1,14}([-\s]?\d{1,13})?', #+xx xxxxxxxxx
        'Tracking': r'\b1Z[A-Z0-9]{16}\b|\b(?!\+\d{11,14}\b)\d{12,15}\b',  # USP OR fedEx tracking
        # r'(?<=\s)\d{4}-\d{4}-\d{4}-\d{4}(?=\s)' can't specifically find any for credit card
        'Postal': r'\b[ABCEGHJKLMNPRSTVXY]\d[ABCEGHJKLMNPRSTVXY] ?\d[ABCEGHJKLMNPRSTVXY]\d\b'  # postal code
    }
    data['Amount'] = 0
    data['Tracking'] = 0
    data['Postal'] = 0
    phishing['Domestic_Phone'] = 0
    phishing['International_Phone'] = 0

    def flag_feature(text, pattern):
        return 1 if re.search(pattern, text) else 0

    for feature, pattern in all_patterns.items():
        data[feature] = (data['Subject'].apply(lambda x: flag_feature(x, pattern)) |
                         data['Body'].apply(lambda x: flag_feature(x, pattern)))

    return data[['Amount', 'Tracking', 'Postal', 'Domestic_Phone', 'International_Phone']]

#testing
extracted = extract_features(phishing)
flagged = extracted[(extracted['Amount'] == 1) | (extracted['Tracking'] == 1) | (extracted['Postal'] == 1)
                    | (extracted['Domestic_Phone'] == 1) | (extracted['International_Phone'] == 1)]
#print(flagged[['Subject','Body']])
print(flagged.shape[0])
#print(flagged[['Amount','Tracking','Postal','Domestic_Phone','International_Phone']])
extracted.to_csv('abnormal_number.csv', index=False)