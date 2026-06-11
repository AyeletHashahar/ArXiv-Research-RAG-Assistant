from datasets import load_dataset
import pandas as pd
import os
from itertools import islice

def main():
    os.makedirs("data", exist_ok=True)

    dataset = load_dataset(
        "ccdv/arxiv-summarization",
        split="train",
        streaming=True
    )

    samples = list(islice(dataset, 1000))
    rows = []
    for i, sample in enumerate(samples):
        rows.append({
            "id": i,
            "article": sample["article"],
            "abstract": sample["abstract"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/arxiv-summarization.csv", index=False)

    print(f"Saved {len(df)} rows to data/arxiv-summarization.csv")

def load_data():
    df = pd.read_csv("data/arxiv-summarization.csv")
    return df

def dataset_explore(df):
    print(df.head())
    print(df.columns)
    print(df.shape)

    print(len(df["article"][0]))
    print(len(df["abstract"][0]))

    print(len(df["article"][10]))
    print(len(df["abstract"][10]))

    print(len(df["article"][100]))
    print(len(df["abstract"][100]))

    print(f"Average length of article: {df['article'].apply(len).mean()}")
    print(f"Average length of abstract: {df['abstract'].apply(len).mean()}")
    print(f"Max length of article: {df['article'].apply(len).max()}")
    print(f"Max length of abstract: {df['abstract'].apply(len).max()}")
    print(f"Min length of article: {df['article'].apply(len).min()}")
    print(f"Min length of abstract: {df['abstract'].apply(len).min()}")



if __name__ == "__main__":
    # main()
    df = load_data()
    dataset_explore(df)