import re
import pandas as pd


TRAIN_CSV_RAW = 'csv/raw/train.csv'
TRAIN_CSV_CLEAN = 'csv/clean/train.csv'

PREDICT_CSV_RAW = 'csv/raw/predict.csv'
PREDICT_CSV_CLEAN = 'csv/clean/predict.csv'

"""
Function to clean and normalize a title string.

This function performs the following operations:
1. Replaces specific characters (hyphen, plus sign, vertical bar, backslash) with spaces.
2. Replaces ampersand (&) with the word "and".
3. Replaces all non-word characters (anything other than letters, digits, and underscores) with spaces.
4. Converts all characters to lowercase for uniformity.
5. Removes all punctuation by iterating over the string.punctuation list.
6. Collapses multiple consecutive spaces into a single space and strips leading and trailing spaces.
"""
def clean(tweet: str):
    try:
        tweet = re.sub(r"\-"," ",tweet)
        tweet = re.sub(r"\+"," ",tweet)
        tweet = re.sub (r"&","and",tweet)
        tweet = re.sub(r"\|"," ",tweet)
        tweet = re.sub(r"\\"," ",tweet)
        tweet = re.sub(r"\W"," ",tweet)
        tweet = re.sub(r"\s+", " ", tweet).strip()
        return tweet.lower()   
    except Exception as e:
        print(f"Input string: {tweet}")
        print(f"An exception occurred: {e}")
        return ""


if __name__ == "__main__":

    
    df = pd.read_csv(TRAIN_CSV_RAW, low_memory=False)
    print(f"Reading: {TRAIN_CSV_RAW}. Number of entries before cleaning: {len(df)}")
    df['tweet'] = df['tweet'].apply(clean)
    print(f"Saving to: {TRAIN_CSV_CLEAN}. Number of entries after cleaning: {len(df)}")
    df.to_csv(TRAIN_CSV_CLEAN, index=False)
    
    df = pd.read_csv(PREDICT_CSV_RAW, low_memory=False)
    print(f"Reading: {PREDICT_CSV_RAW}. Number of entries before cleaning: {len(df)}")
    df['tweet'] = df['tweet'].apply(clean)
    print(f"Saving to: csv/raw/predict_N.csv. Number of entries after cleaning: {len(df)}")
    df.to_csv(PREDICT_CSV_CLEAN, index=False)
    