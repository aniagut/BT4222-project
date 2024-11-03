import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from tqdm import tqdm
import time
start = time.time()
# Paths and configuration
print("loading dataset")
dataset_type = "ebnerd_demo"
base_path = os.path.join(".", dataset_type)
train_path = os.path.join(base_path, "train")
behaviors_path = os.path.join(train_path, "behaviors.parquet")
history_path = os.path.join(train_path, "history.parquet")
articles_path = os.path.join(base_path, "articles.parquet")


# Column selections
articles_columns = ["article_id", "premium", "category", "subcategory", "sentiment_score", "sentiment_label"]
behaviors_columns = ["impression_id", "read_time",
                     "device_type", "article_ids_inview",
                     "impression_time", "article_ids_clicked",
                     "user_id", "is_sso_user",
                     "is_subscriber", "session_id"]

# Load data
behaviors = pd.read_parquet(behaviors_path)[behaviors_columns]
history = pd.read_parquet(history_path)
articles = pd.read_parquet(articles_path)[articles_columns]
embeddings_df = pd.read_parquet('./article_embeddings.parquet')

############################
# CATEGORICAL VARIABLES
############################
category_encoder = LabelEncoder()
subcategory_encoder = LabelEncoder()

articles['category_encoded'] = category_encoder.fit_transform(articles['category'])

# If 'subcategory' is a NumPy array of integers, you can directly encode it
# If it is a list of arrays, you may want to flatten it or handle it differently.
articles['subcategory_encoded'] = subcategory_encoder.fit_transform(
    articles['subcategory'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, np.ndarray) else x))




# Explode arrays in `history` to get individual article impressions
history_exploded = history.explode(['article_id_fixed',
                                    'impression_time_fixed',
                                    'scroll_percentage_fixed',
                                    'read_time_fixed'])

# Rename columns for clarity
history_exploded = history_exploded.rename(columns={
    "article_id_fixed": "article_id",
    "impression_time_fixed": "impression_time",
    "scroll_percentage_fixed": "scroll_percentage",
    "read_time_fixed": "read_time"
})

# Join with articles dataset to get additional features
history_exploded = history_exploded.merge(
    articles[['article_id', 'sentiment_label', 'category_encoded']],
    on='article_id',
    how='left'
)

# Calculate user-level metrics
user_read_time_avg = history_exploded.groupby('user_id')['read_time'].mean().reset_index(name='average_read_time')
user_scroll_avg = history_exploded.groupby('user_id')['scroll_percentage'].mean().reset_index(name='average_scroll_percentage')

# Merge user-level metrics into exploded history
history_exploded = history_exploded.merge(user_read_time_avg, on='user_id', how='left')
history_exploded = history_exploded.merge(user_scroll_avg, on='user_id', how='left')

# Define function to calculate impression frequency (average time between consecutive impressions)
def calculate_impression_frequency(impression_times):
    if len(impression_times) < 2:
        return 0
    time_diffs = np.diff(impression_times).astype('timedelta64[s]')
    return np.mean(time_diffs)

# Apply impression frequency calculation per user
history_exploded["impression_frequency"] = history_exploded.groupby('user_id')['impression_time'].transform(
    lambda x: calculate_impression_frequency(x.values) if x.count() > 1 else 0
)
history_exploded["impression_frequency"] = history_exploded["impression_frequency"].dt.total_seconds()

# Calculate favorite and least favorite categories per user
category_counts = history_exploded.groupby(['user_id', 'category_encoded']).size().reset_index(name='count')

# Favorite category
favorite_category = category_counts.loc[category_counts.groupby('user_id')['count'].idxmax()]
history_exploded = history_exploded.merge(
    favorite_category[['user_id', 'category_encoded']],
    on='user_id',
    how='left',
    suffixes=('', '_favorite')
).rename(columns={"category_encoded_favorite": "favorite_category_encoded"})

# Least favorite category
least_favorite_category = category_counts.loc[category_counts.groupby('user_id')['count'].idxmin()]
history_exploded = history_exploded.merge(
    least_favorite_category[['user_id', 'category_encoded']],
    on='user_id',
    how='left',
    suffixes=('', '_least_favorite')
).rename(columns={"category_encoded_least_favorite": "least_favorite_category_encoded"})

# Calculate interaction score
history_exploded['interaction_score'] = (
    history_exploded['average_read_time'] + history_exploded['average_scroll_percentage']
) / 2

# Calculate the dominant sentiment label for each user
dominant_mood = history_exploded.groupby('user_id')['sentiment_label'].agg(lambda x: x.value_counts().idxmax()).reset_index(name='user_mood')

# Merge user mood into exploded history
history_exploded = history_exploded.merge(dominant_mood, on='user_id', how='left')

# Merge features back into the original history DataFrame to get one value per user
history_FE = history_exploded.groupby('user_id').agg({
    'average_read_time': 'mean',
    'average_scroll_percentage': 'mean',
    'impression_frequency': 'mean',
    'favorite_category_encoded': 'first',
    'least_favorite_category_encoded': 'first',
    'interaction_score': 'mean',
    'user_mood': 'first'
}).reset_index()




###########################
############################
# SPLITTING AND MERGING
############################




val_date = '2023-05-23 07:00:00'

# Filter train/test
behaviors_train = behaviors[behaviors["impression_time"] <= val_date]
behaviors_test = behaviors[behaviors["impression_time"] > val_date]

# Pre-process behaviors for merging
behaviors_train = behaviors_train.explode("article_ids_inview").reset_index(drop=True)
behaviors_train["article_ids_clicked"] = behaviors_train["article_ids_clicked"].apply(
    lambda x: list(map(int, x)) if isinstance(x, (list, np.ndarray)) else []
)
behaviors_train["clicked"] = behaviors_train.apply(lambda x: int(x["article_ids_inview"]) in x["article_ids_clicked"], axis=1)


"""
we can merge later also embeddings by article_id

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
    history_FE,
    left_on="user_id",
    right_on="user_id",
    how="left"
)


# Select and rename columns
final_data = merged_data
print("saving to parquet")
# Save to CSV
file_name = f"xgboost_dataset_{dataset_type}"
final_data.to_parquet(f"{file_name}.parquet", index=False)
end = time.time()
print(f"Saved {file_name} of length: {len(final_data)} in {end-start:.2f} seconds.")

# Preview





features_cont = ['read_time', 'device_type',
                 'is_sso_user', 'is_subscriber', 'premium',
                 'sentiment_score', 'average_read_time',
                 'average_scroll_percentage', 'impression_frequency',
                 'interaction_score']

features_cat = ['sentiment_label', 'user_mood', 'category_encoded', 'subcategory_encoded', 'favorite_category_encoded', 'least_favorite_category_encoded']
