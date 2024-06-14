import pandas as pd


def extract_subject_from_message(message):
    lines = message.split('\n')  # Split the message into a list by row
    subject = ''  # Initialize the subject string
    start_index = -1  # Initialize the start index to -1
    for i, line in enumerate(lines):
        if 'Subject:' in line:  # If the current line contains 'Subject:', indicates the start of the subject
            start_index = i  # Record the row index at the beginning of the subject
            break
    if start_index != -1:  # If the opening line of the subject is found
        for i in range(start_index, len(lines)):  # Walk through the lines starting with the subject
            if 'Content-Type:' in lines[
                i]:  # If the current line contains 'Content-Type:', indicates the end of the subject
                break
            subject += lines[i].replace('Subject:', '').strip() + ' '  # Adds the current line to the subject string
    return subject.strip()  # Returns the subject string with whitespace removed


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


def create_combined_csv(normal_csv_path, phishing_csv_path, output_csv_path):
    # Read the CSV files
    normal_df = pd.read_csv(normal_csv_path)
    phishing_df = pd.read_csv(phishing_csv_path)

    # Remove 'file' column from normal emails
    normal_df = normal_df.drop(columns=['file'])

    # Extract subject and body from normal emails
    normal_df['Subject'] = normal_df['message'].apply(extract_subject_from_message)
    normal_df['Body'] = normal_df['message'].apply(extract_body_from_message)

    # Drop the original 'message' column from normal emails
    normal_df = normal_df.drop(columns=['message'])

    # Add label columns
    normal_df['label'] = 1
    phishing_df['label'] = 0

    # Rename columns in phishing emails to match the normal emails
    phishing_df = phishing_df.rename(columns={'message': 'Body'})

    # Select necessary columns in the same order
    normal_df = normal_df[['Subject', 'Body', 'label']]
    phishing_df = phishing_df[['Subject', 'Body', 'label']]

    # Count the number of rows in phishing email dataset
    num_phishing = len(phishing_df)

    # Calculate the number of rows to randomly select from normal email dataset
    num_normal_to_select = num_phishing * 10

    # Randomly select rows from normal email dataset
    normal_selected_df = normal_df.sample(n=num_normal_to_select, random_state=1)

    # Combine the selected normal email rows with phishing email rows
    combined_df = pd.concat([normal_selected_df, phishing_df])

    # Save the combined dataset to a new CSV file
    combined_df.to_csv(output_csv_path, index=False)
    # # Remove 'Unnamed: 2' and 'Unnamed: 3' columns
    # normal_df = normal_df.drop(columns=['Unnamed: 2', 'Unnamed: 3'])
    # phishing_df = phishing_df.drop(columns=['Unnamed: 2', 'Unnamed: 3'])

# Use the function
normal_csv_path = 'emails.csv'  # Replace with the path to your normal email CSV file
phishing_csv_path = 'CaptstoneProjectData_2024.csv'  # Replace with the path to your phishing email CSV file
output_csv_path = 'Combined_email.csv'  # Replace with the desired output CSV file path

create_combined_csv(normal_csv_path, phishing_csv_path, output_csv_path)
