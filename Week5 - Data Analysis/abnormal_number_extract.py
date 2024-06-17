import re
import pandas as pd
#### takes ony subject & body as individual data
def extract_abnormal_number(data):
    # define patterns
    all_patterns = {
        'Amount': r'\$\s?\d+\.?\d*',  # CAD dollar
        'Domestic_Phone': r'\b(\+1[-\s]?)?\d{3}[-\s]\d{3}[-\s]\d{4}\b',  # +1 | 1 | nothing + xxx xxx xxxx or xxx-xxx-xxxx
        'International_Phone': r'\+(?!1)\d{1,3}\s?\d{1,14}([-\s]?\d{1,13})?', # +xx xxxxxxxxx
        'Tracking': r'\b1Z[A-Z0-9]{16}\b|\b(?!\+\d{11,14}\b)\d{12,15}\b',  # USP OR FedEx tracking
        'Postal': r'\b[ABCEGHJKLMNPRSTVXY]\d[ABCEGHJKLMNPRSTVXY] ?\d[ABCEGHJKLMNPRSTVXY]\d\b'  # postal code
    }

    #def flag_feature(text, pattern):
    #    return 1 if re.search(pattern, text) else 0

    #for feature in all_patterns.keys():
    #    data[feature] = 0

    #for feature, pattern in all_patterns.items():
    #  data[feature] = (data['Subject'].apply(lambda x: flag_feature(x, pattern)) |
    #                    data['Body'].apply(lambda x: flag_feature(x, pattern)))

    #return data[['Amount', 'Tracking', 'Postal', 'Domestic_Phone', 'International_Phone']]

    def count_pattern_occurrences(text, pattern):
        return len(re.findall(pattern, text))

    for feature in all_patterns.keys():
        data[feature] = 0

    for feature, pattern in all_patterns.items():
        data[feature] = data['Subject'].apply(lambda x: count_pattern_occurrences(x, pattern)) + \
                        data['Body'].apply(lambda x: count_pattern_occurrences(x, pattern))

    # Summing up all the counts into a single column
    data['Total_Abnormal_Count'] = data[list(all_patterns.keys())].sum(axis=1)

    return data[['Total_Abnormal_Count']]
