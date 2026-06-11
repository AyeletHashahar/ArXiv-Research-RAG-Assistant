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



if __name__ == "__main__":
    main()