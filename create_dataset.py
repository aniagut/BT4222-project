import pandas as pd
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

class ArticlesDataset(Dataset):
    def __init__(self, articles):
        self.articles = articles

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, article_id):
        article_row = self.articles.iloc[self.articles['article_id'] == article_id]

        return {
            'article_id': torch.tensor(article_row['article_id'], dtype=torch.long),
            'category': torch.tensor(article_row['category'], dtype=torch.long),
            'sentiment_score': torch.tensor(article_row['sentiment_score'], dtype=torch.float),
            'premium': torch.tensor(article_row['premium'], dtype=torch.bool)
        }

# Create DataLoader for the articles dataset
batch_size = 512
articles_dataset = ArticlesDataset(articles_limit)
articles_loader = DataLoader(articles_dataset, batch_size=batch_size, shuffle=True)

class BehaviorsDataset(Dataset):
    def __init__(self, behaviors):
        self.behaviors = behaviors

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        behavior_row = self.behaviors.iloc[idx]

        return {
            'impression_id': torch.tensor(behavior_row['impression_id'], dtype=torch.long),
            'article_id': torch.tensor(behavior_row['article_id'], dtype=torch.long),
            'read_time': torch.tensor(behavior_row['read_time'], dtype=torch.float),
            'scroll_percentage': torch.tensor(behavior_row['scroll_percentage'], dtype=torch.float),
            'device_type': torch.tensor(behavior_row['device_type'], dtype=torch.long),
            'article_ids_inview': torch.tensor(behavior_row['article_ids_inview'], dtype=torch.long),
            'article_ids_clicked': torch.tensor(behavior_row['article_ids_clicked'], dtype=torch.long),
            'user_id': torch.tensor(behavior_row['user_id'], dtype=torch.long),
            'is_sso_user': torch.tensor(behavior_row['is_sso_user'], dtype=torch.bool),
            'is_subscriber': torch.tensor(behavior_row['is_subscriber'], dtype=torch.bool),
            'session_id': torch.tensor(behavior_row['session_id'], dtype=torch.long),
        }


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
        history_row = self.history.iloc[self.history['user_id'] == user_id]

        return {
            'user_id': torch.tensor(history_row['user_id'], dtype=torch.long),
            'impression_time_fixed': torch.tensor(history_row['impression_time_fixed'], dtype=torch.float),
            'scroll_percentage_fixed': torch.tensor(history_row['scroll_percentage_fixed'], dtype=torch.float),
            'article_id_fixed': torch.tensor(history_row['article_id_fixed'], dtype=torch.long),
            'read_time_fixed': torch.tensor(history_row['read_time_fixed'], dtype=torch.float),
        }

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

    def __getitem__(self, impression_id):
        impression_row = self.behaviors_dl[impression_id]
        article_inviews = impression_row["article_ids_inview"]
        article_tensors_list = []
        for article_inview in article_inviews:
            article_row = self.articles_dl[article_inview]

            # Convert the article row to a tensor
            article_tensor = torch.tensor(article_row.values, dtype=torch.float)  # Adjust dtype as needed
            article_tensors_list.append(article_tensor)  # Append the article tensor to the list

        # Concatenate all collected article tensors into a single tensor
        concatenated_articles = torch.stack(article_tensors_list)

        user_id = impression_row["user_id"]
        articles_history_user_id = self.history_dl[user_id]["article_id_fixed"]




# # Set model hyperparameters: latent dimension and number of layers
# latent_dim = 64  # Dimensionality of the latent feature space
# n_layers = 3  # Number of layers in the neural network
#
# import torch
#
# # Define a function to convert a DOK matrix to a sparse tensor in PyTorch
# def convert_to_sparse_tensor(dok_mtrx):
#     dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)  # Convert DOK matrix to COO format and ensure data type
#     values = dok_mtrx_coo.data  # Extract non-zero values
#     indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))
#
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = dok_mtrx_coo.shape
#
#     dok_mtrx_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
#
#     return dok_mtrx_sparse_tensor


