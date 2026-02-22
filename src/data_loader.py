import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path: str, keyword: str = "climate", sample_size: int = 10000):
    df = pd.read_csv(file_path, encoding="latin-1", header=None)
    df.columns = ["sentiment", "id", "date", "query", "user", "text"]

    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    df = df[df["text"].str.contains(keyword, case=False, na=False)]

    return df[["sentiment", "text"]]