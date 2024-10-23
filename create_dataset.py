import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset

dataset_type = "ebnerd_demo"
base_path = os.path.join(".", dataset_type)
train_path = os.path.join(base_path, "train")
behaviors_path = os.path.join(train_path, "behaviors.parquet")
history_path = os.path.join(train_path, "history.parquet")
articles_path = os.path.join(base_path, "articles.parquet")

articles_columns = ["article_id", "premium",
                    "category",
                    "sentiment_score"]

behaviors_columns = ["impression_id", "article_id",
                     "read_time", "scroll_percentage",
                     "device_type", "article_ids_inview",
                     "impression_time",
                     "article_ids_clicked", "user_id",
                     "is_sso_user", "is_subscriber",
                     "session_id"]

behaviors = pd.read_parquet(behaviors_path)
history = pd.read_parquet(history_path)
articles = pd.read_parquet(articles_path)

behaviors_limit = behaviors[behaviors_columns]
articles_limit = articles[articles_columns]

# print(behaviors.head())
# print(history.head())
# print(articles.head())

print(behaviors_limit.head())
print(history.head())
print(articles_limit.head())

batch_size = 512

class ArticlesDataset(Dataset):
    def __init__(self, articles):
        self.articles = articles

    def __len__(self):
        return len(self.articles)

    # def __getitem__(self, article_id):
    #     article_row = self.articles.loc[self.articles['article_id'] == article_id]
    #
    #     return {
    #         'article_id': torch.tensor(article_row['article_id'], dtype=torch.long),
    #         'category': torch.tensor(article_row['category'], dtype=torch.long),
    #         'sentiment_score': torch.tensor(article_row['sentiment_score'], dtype=torch.float),
    #         'premium': torch.tensor(article_row['premium'], dtype=torch.bool)
    #     }

    def __getitem__(self, article_id):
        # Get all rows that match the article_id
        article_rows = self.articles.loc[self.articles['article_id'] == article_id]

        # Check if any articles were found
        if article_rows.empty:
            raise ValueError(f"No articles found for article_id: {article_id}")

        # Convert the relevant columns to tensors
        article_data = {
            'article_id': torch.tensor(article_rows['article_id'].values, dtype=torch.long),
            'category': torch.tensor(article_rows['category'].values, dtype=torch.long),
            'sentiment_score': torch.tensor(article_rows['sentiment_score'].values, dtype=torch.float),
            'premium': torch.tensor(article_rows['premium'].values, dtype=torch.bool)
        }

        return article_data

articles_dataset = ArticlesDataset(articles_limit)
articles_loader = DataLoader(articles_dataset, batch_size=batch_size, shuffle=True)

class BehaviorsDataset(Dataset):
    def __init__(self, behaviors):
        self.behaviors = behaviors

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        # behavior_row = self.behaviors[idx]
        behavior_row = self.behaviors.loc[self.behaviors['impression_id'] == idx]

        # return {
        #     'impression_id': torch.tensor(behavior_row['impression_id'], dtype=torch.long),
        #     'article_id': torch.tensor(behavior_row['article_id'], dtype=torch.long),
        #     'read_time': torch.tensor(behavior_row['read_time'], dtype=torch.float),
        #     'scroll_percentage': torch.tensor(behavior_row['scroll_percentage'], dtype=torch.float),
        #     'device_type': torch.tensor(behavior_row['device_type'], dtype=torch.long),
        #     'article_ids_inview': torch.tensor(behavior_row['article_ids_inview'], dtype=torch.long),
        #     'article_ids_clicked': torch.tensor(behavior_row['article_ids_clicked'], dtype=torch.long),
        #     'user_id': torch.tensor(behavior_row['user_id'], dtype=torch.long),
        #     'is_sso_user': torch.tensor(behavior_row['is_sso_user'], dtype=torch.bool),
        #     'is_subscriber': torch.tensor(behavior_row['is_subscriber'], dtype=torch.bool),
        #     'session_id': torch.tensor(behavior_row['session_id'], dtype=torch.long),
        # }
        return behavior_row

behaviors_limit_train = behaviors_limit[behaviors_limit["impression_time"] <='2023-05-23 07:00:00']
behaviors_limit_test = behaviors_limit[behaviors_limit["impression_time"] > '2023-05-23 07:00:00']

# Create DataLoader for the behaviors dataset
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
        history_row = self.history.loc[self.history['user_id'] == user_id]

        return history_row

# Create DataLoader for the history dataset
history_dataset = HistoryDataset(history)
history_train_loader = DataLoader(history_dataset, batch_size=batch_size)

class CombineDataset(DataLoader):
    def __init__(self, articles_dl, behaviors_dl, history_dl):
        self.articles_dl = articles_dl
        self.behaviors_dl = behaviors_dl
        self.history_dl = history_dl

    def __len__(self):
        return len(self.articles_dl) + len(self.behaviors_dl) + len(self.history_dl)

    # def __getitem__(self, impression_id):
    #     impression_row = self.behaviors_dl[impression_id]
    #
    #     # Retrieve the specific columns from the impression row
    #     impression_id_value = impression_row["impression_id"]
    #     article_id_value = impression_row["article_id"]
    #     read_time_value = impression_row["read_time"]
    #     scroll_percentage_value = impression_row["scroll_percentage"]
    #     device_type_value = impression_row["device_type"]
    #     article_inviews = impression_row["article_ids_inview"]
    #     article_ids_clicked_value = impression_row["article_ids_clicked"]
    #     user_id_value = impression_row["user_id"]
    #     is_sso_user_value = impression_row["is_sso_user"]
    #     is_subscriber_value = impression_row["is_subscriber"]
    #     session_id_value = impression_row["session_id"]
    #
    #     # Convert these values into tensors (if necessary)
    #     read_time_tensor = torch.tensor([read_time_value], dtype=torch.float)
    #     scroll_percentage_tensor = torch.tensor([scroll_percentage_value], dtype=torch.float)
    #     device_type_tensor = torch.tensor([device_type_value], dtype=torch.float)  # Assuming device type is numeric
    #
    #
    #
    #     article_tensors_list = []
    #     for article_inview in article_inviews:
    #         article_row = self.articles_dl[article_inview]
    #
    #         # Convert the article row to a tensor
    #         article_tensor = torch.tensor(article_row.values, dtype=torch.float)
    #         article_tensors_list.append(article_tensor)
    #
    #     # Concatenate all collected article tensors into a single tensor
    #     concatenated_inview_articles = torch.stack(article_tensors_list)
    #
    #     # Tensor of history articles
    #     user_id = impression_row["user_id"]
    #     articles_history_user_id = self.history_dl[user_id]["article_id_fixed"]
    #     articles_user_id_tensor = []
    #     for article_history in articles_history_user_id:
    #         article_row = self.articles_dl[article_history]
    #
    #         article_tensor = torch.tensor(article_row.values, dtype=torch.float)
    #         articles_user_id_tensor.append(article_tensor)
    #
    #     # Concatenate all collected article tensors into a single tensor
    #     concatenated_history_articles = torch.stack(articles_user_id_tensor)
    #
    #
    #
    #     # Combine all the pieces of data into a single dictionary or list
    #     combined_data = {
    #         "impression_id": torch.tensor([impression_id_value], dtype=torch.int),
    #         "article_id": torch.tensor([article_id_value], dtype=torch.int),
    #         "article_ids_inview": concatenated_inview_articles,
    #         "article_ids_clicked": torch.tensor(article_ids_clicked_value, dtype=torch.int),
    #         "user_id": torch.tensor([user_id_value], dtype=torch.int)
    #         # "session_id": torch.tensor([session_id_value], dtype=torch.int)
    #     }
    #
    #     return combined_data

    def __getitem__(self, impression_id):
        # Get the row from behaviors_dl using the impression_id
        impression_row = self.behaviors_dl[impression_id]

        # Retrieve the specific columns from the impression row
        session_id_value = impression_row["session_id"]
        user_id_value = impression_row["user_id"]
        article_ids_clicked_value = impression_row["article_ids_clicked"]
        article_inviews = impression_row["article_ids_inview"]


        # Create a list to store rows for the DataFrame
        rows = []

        article_inviews_list = article_inviews.tolist()[0]

        # Iterate over each article in view
        for article_inview in article_inviews_list:
            article_row = self.articles_dl[article_inview]

            # Check if the article was clicked
            is_clicked = 1 if int(article_inview) in map(int, article_ids_clicked_value) else 0

            sentiment_score = self.articles_dl[article_inview]["sentiment_score"]
            is_premium_user = self.behaviors_dl[impression_id]["is_subscriber"]
            readtime_avg = np.mean(self.history_dl[user_id_value.tolist()[0]]["read_time_fixed"].tolist()[0])

            # Create a row dictionary with the necessary data
            row = {
                "session_id": session_id_value,
                "user_id": user_id_value,
                "article_id": article_inview,  # Article in view
                "clicked": is_clicked,  # 1 if clicked, 0 otherwise
                "sentiment_score": sentiment_score,
                "is_premium_user": is_premium_user,
                "readtime_avg": readtime_avg
            }

            # Add the row to the list
            rows.append(row)

        # Create a DataFrame from the rows list
        df = pd.DataFrame(rows)

        return df



combineTest = CombineDataset(articles_dataset, behaviors_train_dataset, history_dataset)

#get list of impression ids
list_of_impression_ids_train = behaviors_limit_train["impression_id"].tolist()# Initialize a list to store DataFrames
data_frames = []
# Loop through each impression ID and retrieve the corresponding DataFrame
for impression_id in list_of_impression_ids_train:
    try:
        id_df = combineTest[int(impression_id)]  # This assumes __getitem__ is implemented
    except:
        print(impression_id)
        continue
    data_frames.append(id_df)

# Concatenate all DataFrames into a single DataFrame
concatenated_data = pd.concat(data_frames, ignore_index=True)
print(concatenated_data.head())
concatenated_data.to_csv("xgboost_dataset.csv", index=False)


# concatenate datasets from list of impression ids from the __getitem__ method



# Set model hyperparameters: latent dimension and number of layers
latent_dim = 64  # Dimensionality of the latent feature space
n_layers = 3  # Number of layers in the neural network

# Define a function to convert a DOK matrix to a sparse tensor in PyTorch
def convert_to_sparse_tensor(dok_mtrx):
    dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)  # Convert DOK matrix to COO format and ensure data type
    values = dok_mtrx_coo.data  # Extract non-zero values
    indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = dok_mtrx_coo.shape

    dok_mtrx_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return dok_mtrx_sparse_tensor


