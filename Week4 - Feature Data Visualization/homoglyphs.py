from confusable_homoglyphs import confusables
import re
import pandas as pd
import os


def clean_homoglyphs(text):
    # Replace various types of whitespace with a single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def count_homoglyphs_scripts(text):
    return sum(confusables.is_dangerous(word) for word in text.split())


def feature_homoglyphs(data):
    data["Subject"] = data["Subject"].apply(clean_homoglyphs)
    data["Body"] = data["Body"].apply(clean_homoglyphs)
    data["Homoglyphs Scripts Count"] = data.apply(
        lambda row: count_homoglyphs_scripts(row["Subject"] + " " + row["Body"]), axis=1
    )
    data.drop(columns=["Subject", "Body"], inplace=True)
    return data[["Homoglyphs Scripts Count"]]


def normal_extract_subject_body(text):
    # Extract the subject using a regular expression
    subject_match = re.search(r"Subject: (.*)", text)
    subject = subject_match.group(1) if subject_match else "Subject Not Found"

    # Clean the text by removing everything from "Message-ID:" up to "X-FileName:"
    cleaned_text = re.sub(r"Message-ID:.*?X-FileName:.*?\n", "", text, flags=re.S)

    return subject, cleaned_text


def main():
    project_root = "/Users/feiyixie/Projects/Summer-2024-ECE-597-Group8"

    phishing_data_path = os.path.join(
        project_root, "data", "raw", "CaptstoneProjectData_2024.csv"
    )
    normal_data_path = os.path.join(
        project_root, "data", "raw", "EnronEmailDataset.csv"
    )

    phishing_data = pd.read_csv(phishing_data_path)
    normal_data = pd.read_csv(
        normal_data_path, nrows=10000
    )  # Read only 5000 rows of normal data for now

    phishing_data.drop(columns=["Unnamed: 2", "Unnamed: 3"], inplace=True)
    phishing_data.dropna(
        subset=["Subject", "Body"], inplace=True
    )  # Remove rows with missing subject or body
    normal_data[["Subject", "Body"]] = normal_data["message"].apply(
        lambda x: pd.Series(normal_extract_subject_body(x))
    )

    df_phishing_data = feature_homoglyphs(phishing_data)
    df_normal_data = feature_homoglyphs(normal_data)
    print("Phishing Data")
    print(df_phishing_data.head())
    print("Normal Data")
    print(df_normal_data.head())


if __name__ == "__main__":
    main()
