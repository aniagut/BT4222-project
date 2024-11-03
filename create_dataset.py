import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Paths and configuration
dataset_type = "ebnerd_small"
base_path = os.path.join(".", dataset_type)
train_path = os.path.join(base_path, "train")
behaviors_path = os.path.join(train_path, "behaviors.parquet")
history_path = os.path.join(train_path, "history.parquet")
articles_path = os.path.join(base_path, "articles.parquet")

articles_columns = ["article_id", "premium", "category", "sentiment_score", "body"]

behaviors_columns = ["impression_id", "article_id", "read_time", "scroll_percentage", "device_type",
                     "article_ids_inview", "impression_time",
                     "article_ids_clicked", "user_id",
                     "is_sso_user", "is_subscriber",
                     "session_id"]

behaviors = pd.read_parquet(behaviors_path)
history = pd.read_parquet(history_path)
articles = pd.read_parquet(articles_path)


behaviors_limit = behaviors[behaviors_columns]
articles_limit = articles[articles_columns]

print(behaviors_limit.head())
print(history.head())
print(articles_limit.head())

batch_size = 512


class BehaviorsDataset(Dataset):
    def __init__(self, behaviors):
        self.behaviors = behaviors

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        return self.behaviors.loc[self.behaviors['impression_id'] == idx]

# Split behaviors into training and test sets
behaviors_limit_train = behaviors_limit[behaviors_limit["impression_time"] <='2023-05-23 07:00:00']
behaviors_limit_test = behaviors_limit[behaviors_limit["impression_time"] > '2023-05-23 07:00:00']

behaviors_train_dataset = BehaviorsDataset(behaviors_limit_train)
behaviors_test_dataset = BehaviorsDataset(behaviors_limit_test)

behaviors_train_loader = DataLoader(behaviors_train_dataset, batch_size=batch_size, shuffle=True)
behaviors_test_loader = DataLoader(behaviors_test_dataset, batch_size=batch_size)

class HistoryDataset(Dataset):
    def __init__(self, history):
        self.history = history

    def __len__(self):
        return len(self.history)

    def __getitem__(self, user_id):
        return self.history.loc[self.history['user_id'] == user_id]

history_dataset = HistoryDataset(history)
history_train_loader = DataLoader(history_dataset, batch_size=batch_size)

class CombineDataset(DataLoader):
    def __init__(self, articles_dl, behaviors_dl, history_dl):
        self.articles_dl = articles_dl
        self.behaviors_dl = behaviors_dl
        self.history_dl = history_dl

    def __len__(self):
        return len(self.articles_dl) + len(self.behaviors_dl) + len(self.history_dl)

    def __getitem__(self, impression_id):
        impression_row = self.behaviors_dl[impression_id]

        session_id_value = impression_row["session_id"].values[0]
        user_id_value = impression_row["user_id"].values[0]
        article_ids_clicked_value = impression_row["article_ids_clicked"].values[0]
        article_inviews = impression_row["article_ids_inview"].values[0]

        rows = []
        for article_inview in article_inviews:
            article_row = self.articles_dl[article_inview]
            # Check if the article was clicked
            is_clicked = 1 if int(article_inview) in map(int, article_ids_clicked_value) else 0
            sentiment_score = self.articles_dl[article_inview]["sentiment_score"].numpy()[0]
            is_premium_user = self.behaviors_dl[impression_id]["is_subscriber"].tolist()[0]
            # TODO to fix from series and remove toList
            readtime_avg = np.mean(self.history_dl[user_id_value]["read_time_fixed"].tolist()[0])

            row = {
                "session_id": session_id_value,
                "user_id": user_id_value,
                "article_id": article_inview,
                "clicked": is_clicked,
                "sentiment_score": sentiment_score,
                "is_premium_user": is_premium_user,
                "readtime_avg": readtime_avg
            }

            rows.append(row)

        return pd.DataFrame(rows)

combineTest = CombineDataset(articles_dataset, behaviors_train_dataset, history_dataset)

list_of_impression_ids_train = behaviors_limit_train["impression_id"].tolist()
data_frames = []

for impression_id in tqdm(list_of_impression_ids_train):
    try:
        id_df = combineTest[int(impression_id)]
    except Exception as e:
        print(f"Error with impression_id {impression_id}: {e}")
        continue
    data_frames.append(id_df)

concatenated_data = pd.concat(data_frames, ignore_index=True)
print(concatenated_data.head())
print(len(concatenated_data))
concatenated_data.to_csv(f"xgboost_dataset_{dataset_type}.csv", index=False)

