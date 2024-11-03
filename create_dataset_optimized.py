import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import time
start = time.time()
# Paths and configuration
print("loading dataset")
dataset_type = "ebnerd_small"
base_path = os.path.join(".", dataset_type)
train_path = os.path.join(base_path, "train")
behaviors_path = os.path.join(train_path, "behaviors.parquet")
history_path = os.path.join(train_path, "history.parquet")
articles_path = os.path.join(base_path, "articles.parquet")

# Column selections
articles_columns = ["article_id", "premium", "category", "sentiment_score"]
behaviors_columns = ["impression_id", "article_id", "read_time", "scroll_percentage", "device_type",
                     "article_ids_inview", "impression_time", "article_ids_clicked", "user_id",
                     "is_sso_user", "is_subscriber", "session_id"]

# Load data
behaviors = pd.read_parquet(behaviors_path)[behaviors_columns]
history = pd.read_parquet(history_path)
articles = pd.read_parquet(articles_path)[articles_columns]

# Filter train/test
behaviors_train = behaviors[behaviors["impression_time"] <= '2023-05-23 07:00:00']
behaviors_test = behaviors[behaviors["impression_time"] > '2023-05-23 07:00:00']

# Pre-process behaviors for merging
behaviors_train = behaviors_train.explode("article_ids_inview").reset_index(drop=True)
behaviors_train["article_ids_clicked"] = behaviors_train["article_ids_clicked"].apply(
    lambda x: list(map(int, x)) if isinstance(x, (list, np.ndarray)) else []
)
behaviors_train["clicked"] = behaviors_train.apply(lambda x: int(x["article_ids_inview"]) in x["article_ids_clicked"], axis=1)

# Pre-calculate average read times for each user in history
user_readtime_avg = history.groupby("user_id")["read_time_fixed"].mean().rename("readtime_avg")
history["readtime_avg"] = history["read_time_fixed"].apply(lambda x: np.mean(x) if isinstance(x, np.ndarray) else 0)
user_readtime_avg = history["readtime_avg"]

"""
we can merge later also embeddings by article_id
embeddings_df = pd.read_parquet('./article_embeddings.parquet')
articles_emb = pd.merge(
    articles,
    embeddings_df,
    left_on="article_id",
    right_index=True,
    how="left"
)

and next merge should be (instead of existing merge_data)

merged_data = pd.merge(
    behaviors_train,
    articles_emb,
    left_on="article_ids_inview",
    right_on="article_id",
    how="left"
)
"""

print("Merging behaviour and articles")
# Merge data
# 1. Merge behaviors and articles
merged_data = pd.merge(
    behaviors_train,
    articles,
    left_on="article_ids_inview",
    right_on="article_id",
    how="left"
)

print("Merging user and read_time")
# 2. Merge user read time
merged_data = pd.merge(
    merged_data,
    user_readtime_avg,
    left_on="user_id",
    right_index=True,
    how="left"
).fillna({'readtime_avg': 0})  # Fill missing readtime with 0 if no history is available

# Select and rename columns
final_data = merged_data[[
    "session_id", "user_id", "article_ids_inview", "clicked", "sentiment_score",
    "is_subscriber", "readtime_avg"
]].rename(columns={
    "article_ids_inview": "article_id",
    "is_subscriber": "is_premium_user"
})
print("saving to parquet")
# Save to CSV
file_name = f"xgboost_dataset_{dataset_type}"
final_data.to_parquet(f"{file_name}.parquet", index=False)
end = time.time()
print(f"Saved {file_name} of length: {len(final_data)} in {end-start:.2f} seconds.")

# Preview

#print(len(final_data))
