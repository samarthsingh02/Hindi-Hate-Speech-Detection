import pandas as pd
import os
import re
import emoji

def clean_text(text):
    text = str(text)
    # 1. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # 2. Remove @usernames
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # 3. Remove Emojis
    text = emoji.replace_emoji(text, replace='')
    # 4. Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Configuration & Paths ---
# Update these filenames to match exactly what is in your data/raw folder.
# If they are excel files, change .csv to .xlsx and use pd.read_excel below.
FILE_PATH_SUPERSET = '../data/raw/india_hate_speech_superset.csv'
FILE_PATH_INDO = '../data/raw/indo_hate_speech_dataset.xlsx'
FILE_PATH_SYNTHETIC = '../data/raw/code_mixed_hinglish_synthetic.csv'

OUTPUT_PATH = '../data/processed/unified_dataset.csv'


def load_and_standardize_superset(filepath):
    print("Loading India Hate Speech Superset...")
    # Using utf-8 to handle Devanagari script properly
    df = pd.read_csv(filepath, encoding='utf-8')
    # Select and rename columns to standard 'text' and 'label'
    df = df[['text', 'labels']].rename(columns={'labels': 'label'})
    return df


def load_and_standardize_indo(filepath):
    print("Loading Indo-HateSpeech Dataset...")
    df = pd.read_excel(filepath)
    df = df[['Comment', 'Label']].rename(columns={'Comment': 'text', 'Label': 'label'})

    # Clean the labels: Remove potential quotes (e.g., 'HS0') and whitespace
    df['label'] = df['label'].astype(str).str.strip("' ")

    # [cite_start]Map to Binary (Hate / Non-hate) [cite: 2]
    label_mapping = {
        'HS0': 0,  # Non-Hate
        'HS1': 1,  # Hate
        'HSN': 1  # Extreme Hate -> mapped to Hate for binary classification
    }
    df['label'] = df['label'].map(label_mapping)
    return df

def load_and_standardize_synthetic(filepath):
    print("Loading Code-Mixed Hinglish Synthetic Dataset...")
    df = pd.read_csv(filepath, encoding='utf-8')
    df = df[['text', 'hate_label']].rename(columns={'hate_label': 'label'})
    return df


def main():
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    try:
        # 1. Load and standardize individual datasets
        df_superset = load_and_standardize_superset(FILE_PATH_SUPERSET)
        df_indo = load_and_standardize_indo(FILE_PATH_INDO)
        df_synthetic = load_and_standardize_synthetic(FILE_PATH_SYNTHETIC)

        # 2. Merge into a unified dataset
        print("\nMerging datasets...")
        unified_df = pd.concat([df_superset, df_indo, df_synthetic], ignore_index=True)

        # 3. Basic Cleaning & Normalization
        print("Cleaning unified data...")
        # Clean the text column
        print("Applying text cleaning (removing URLs, mentions, emojis)...")
        unified_df['text'] = unified_df['text'].apply(clean_text)
        # Drop rows with missing text or labels
        unified_df.dropna(subset=['text', 'label'], inplace=True)
        # Ensure labels are integers
        unified_df['label'] = unified_df['label'].astype(int)
        # Remove exact duplicates to prevent data leakage
        unified_df.drop_duplicates(subset=['text'], inplace=True)

        # Shuffle the dataset thoroughly
        unified_df = unified_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 4. Save the result
        unified_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
        print(f"\nSuccess! Unified dataset saved to: {OUTPUT_PATH}")
        print(f"Total records: {len(unified_df)}")
        print("Class distribution:")
        print(unified_df['label'].value_counts(normalize=True))

    except FileNotFoundError as e:
        print(f"\nError: Could not find a file. Please check the filenames at the top of the script. Details: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()