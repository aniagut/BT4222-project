import pandas as pd
import os

dataset_type = "ebnerd_demo"
base_path = os.path.join(".", dataset_type)
train_path = os.path.join(base_path, "train")
behaviors_path = os.path.join(train_path, "behaviors.parquet")
history_path = os.path.join(train_path, "history.parquet")
articles_path = os.path.join(base_path, "articles.parquet")

behaviors = pd.read_parquet(behaviors_path)
history = pd.read_parquet(history_path)
articles = pd.read_parquet(articles_path)


print(behaviors.head())
print(history.head())
print(articles.head())