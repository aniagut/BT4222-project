import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from tqdm import tqdm
import time

start = time.time()
# Paths and configuration
print("loading dataset")
dataset_type = "ebnerd_small"
base_path = os.path.join(".", dataset_type)
train_path = os.path.join(base_path, "train")
# validation_path = os.path.join(base_path, "validation")
behaviors_path = os.path.join(train_path, "behaviors.parquet")
# behaviors_path = os.path.join(validation_path, "behaviors.parquet")
history_path = os.path.join(train_path, "history.parquet")
# history_path = os.path.join(validation_path, "history.parquet")
articles_path = os.path.join(base_path, "articles.parquet")

# Column selections
articles_columns = ["article_id",
                    "premium", "category",
                    "subcategory", "sentiment_score",
                    "sentiment_label", "published_time"]

behaviors_columns = ["impression_id",
                     "device_type", "article_ids_inview",
                     "article_ids_clicked",
                     "user_id", "is_sso_user",
                     "is_subscriber", "session_id",
                     # origin article features:
                     "article_id", "impression_time",
                     "read_time", "scroll_percentage"]

# Load data
behaviors = pd.read_parquet(behaviors_path)[behaviors_columns]
behaviors.rename(columns={"article_id": "origin_article_id",
                          "read_time": "origin_read_time",
                          "scroll_percentage": "origin_scroll_percentage"}, inplace=True)

history = pd.read_parquet(history_path)
articles = pd.read_parquet(articles_path)[articles_columns]
embeddings_df = pd.read_parquet('./article_embeddings.parquet')
embedding_clusters = pd.read_parquet('./clustering_results.parquet')
articles = articles.merge(embedding_clusters, left_on="article_id", right_on="article_id", how= "left")

############################
# CATEGORICAL VARIABLES
############################

# Join Behaviors on articles and get the genre o
behaviors_articles = behaviors.merge(articles,
                                     left_on="origin_article_id",
                                     right_on="article_id", how="left")

# Create origin features and fill nan with neutral values for home page
homepage_category = 0
assert homepage_category not in behaviors_articles["category"].values
homepage_id = 0
assert homepage_id not in behaviors_articles["origin_article_id"].values
behaviors["origin_article_id"] = behaviors["origin_article_id"].fillna(homepage_id)
homepage_cluster = np.nanmax(behaviors_articles["cluster"].values)+1
assert homepage_cluster not in behaviors_articles["cluster"].values

# Features
behaviors["coming_from_home_page"] = behaviors["origin_article_id"] == homepage_id
behaviors["origin_cluster"] = behaviors_articles["cluster"].fillna(homepage_cluster)
behaviors["origin_category"] = behaviors_articles["category"].fillna(homepage_category)
behaviors["origin_scroll_percentage"] = behaviors_articles["origin_scroll_percentage"].fillna(0)
behaviors["origin_sentiment_label"] = behaviors_articles["sentiment_label"].fillna("Neutral")
behaviors["origin_sentiment_score"] = behaviors_articles["sentiment_score"].fillna(0.5)
behaviors["origin_published_time"] = behaviors_articles["published_time"].fillna(behaviors["impression_time"])

# WE FILL WITH THE TIME OF THE IMPRESSION, AS THE FRONT PAGE IS ALWAYS UPDATED


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
    articles[['article_id', 'sentiment_label', 'category']],
    on='article_id',
    how='left'
)

# Calculate user-level metrics
user_read_time_avg = history_exploded.groupby('user_id')['read_time'].mean().reset_index(name='user_average_read_time')
user_scroll_avg = history_exploded.groupby('user_id')['scroll_percentage'].mean().reset_index(
    name='user_average_scroll_percentage')

# Merge user-level metrics into exploded history
history_exploded = history_exploded.merge(user_read_time_avg, on='user_id', how='left')
history_exploded = history_exploded.merge(user_scroll_avg, on='user_id', how='left')


# Define function to calculate impression frequency (average time between consecutive impressions)
def calculate_user_impression_frequency(impression_times):
    if len(impression_times) < 2:
        return 0
    time_diffs = np.diff(impression_times).astype('timedelta64[s]')
    return np.mean(time_diffs)


# Apply impression frequency calculation per user
history_exploded["user_impression_frequency"] = history_exploded.groupby('user_id')['impression_time'].transform(
    lambda x: calculate_user_impression_frequency(x.values) if x.count() > 1 else 0
)
history_exploded["user_impression_frequency"] = history_exploded["user_impression_frequency"].dt.total_seconds()

# Calculate favorite and least favorite categories per user
category_counts = history_exploded.groupby(['user_id', 'category']).size().reset_index(name='count')

# Favorite category
favorite_category = category_counts.loc[category_counts.groupby('user_id')['count'].idxmax()]
history_exploded = history_exploded.merge(
    favorite_category[['user_id', 'category']],
    on='user_id',
    how='left',
    suffixes=('', '_favorite')
).rename(columns={"category_favorite": "favorite_category"})

# Least favorite category
least_favorite_category = category_counts.loc[category_counts.groupby('user_id')['count'].idxmin()]
history_exploded = history_exploded.merge(
    least_favorite_category[['user_id', 'category']],
    on='user_id',
    how='left',
    suffixes=('', '_least_favorite')
).rename(columns={"category_least_favorite": "least_favorite_category"})

# Calculate interaction score
history_exploded['user_interaction_score'] = (
                                                     history_exploded['user_average_read_time'] + history_exploded[
                                                 'user_average_scroll_percentage']
                                             ) / 2

# Calculate the dominant sentiment label for each user
dominant_mood = history_exploded.groupby('user_id')['sentiment_label'].agg(
    lambda x: x.value_counts().idxmax()).reset_index(name='user_mood')

# Merge user mood into exploded history
history_exploded = history_exploded.merge(dominant_mood, on='user_id', how='left')

# Merge features back into the original history DataFrame to get one value per user
history_FE = history_exploded.groupby('user_id').agg({
    'user_average_read_time': 'mean',
    'user_average_scroll_percentage': 'mean',
    'user_impression_frequency': 'mean',
    'favorite_category': 'first',
    'least_favorite_category': 'first',
    'user_interaction_score': 'mean',
    'user_mood': 'first'
}).reset_index()

############################
# SPLITTING AND MERGING
############################




# Pre-process behaviors for merging
behaviors = behaviors.explode("article_ids_inview").reset_index(drop=True)
behaviors["article_ids_clicked"] = behaviors["article_ids_clicked"].apply(
    lambda x: int(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan
)
behaviors["clicked"] = behaviors.apply(
    lambda x: int(x["article_ids_inview"]) == x["article_ids_clicked"] if pd.notna(x["article_ids_clicked"]) else False,
    axis=1
)
print("Merging behaviour and articles")
# Merge data
# 1. Merge behaviors and articles
merged_data = pd.merge(
    behaviors,
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

print("User same/diff features")
merged_data["user_article_same_mood"] = merged_data["sentiment_label"] == merged_data["user_mood"]
merged_data["user_article_favorite"] = merged_data["category"] == merged_data["favorite_category"]
merged_data["user_article_least_favorite"] = merged_data["category"] == merged_data["least_favorite_category"]

print("Origin same/diff features")
print("origin features")
merged_data["origin_current_diff_published"] = (
            merged_data["published_time"] - merged_data["origin_published_time"]).dt.total_seconds()
merged_data["origin_current_diff_impression_published"] = (
            merged_data["impression_time"] - merged_data["origin_published_time"]).dt.total_seconds()

merged_data["origin_current_same_cluster"] = merged_data["origin_cluster"] == merged_data["cluster"]
merged_data["origin_current_same_category"] = merged_data["origin_category"] == merged_data["category"]
merged_data["origin_current_same_sentiment_label"] = merged_data["origin_sentiment_label"] == merged_data["sentiment_label"]
merged_data["origin_current_diff_sentiment_score"] = merged_data["origin_sentiment_score"] - merged_data["sentiment_score"]

def categorize_time_of_day(hour):
    if hour < 6:
        return 'Night'
    elif hour < 12:
        return 'Morning'
    elif hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'


merged_data['time_of_day'] = merged_data['impression_time'].dt.hour.apply(categorize_time_of_day)

# Select and rename columns

target = ["clicked"]
ids = ["impression_id", "session_id", "user_id", "article_id"]
feature_user = ['user_average_read_time', 'user_average_scroll_percentage',
                'user_impression_frequency', 'user_interaction_score', 'user_mood', ]
feature_impression = ['device_type', 'is_sso_user', 'is_subscriber', 'origin_read_time',
                      'origin_scroll_percentage', 'coming_from_home_page',
                      'origin_sentiment_label', 'origin_sentiment_score',
                      'origin_current_diff_published',
                      'origin_current_diff_impression_published', 'time_of_day', 'origin_cluster',
                      'origin_current_same_cluster','origin_current_same_category',
                      'origin_current_same_sentiment_label','origin_current_diff_sentiment_score']
feature_article = ['premium', 'sentiment_score', 'sentiment_label', 'user_article_same_mood',
                   'user_article_favorite',
                   'user_article_least_favorite', 'cluster']




val_date = '2023-05-23 07:00:00'
# Filter train/test
merged_data_train = merged_data[merged_data["impression_time"] <= val_date]
merged_data_test = merged_data[merged_data["impression_time"] > val_date]

final_columns = ids + target + feature_user + feature_article + feature_impression

final_data_train = merged_data_train[final_columns]
final_data_test = merged_data_test[final_columns]




# Validation set
print("saving to parquet")
# Save to CSV
file_name_train = f"train_dataset_{dataset_type}"
final_data_train.to_parquet(f"{file_name_train}.parquet", index=False)
print(f"Saved {file_name_train} of length: {len(final_data_train)} ")

file_name_test = f"test_dataset_{dataset_type}"
final_data_test.to_parquet(f"{file_name_test}.parquet", index=False)
print(f"Saved {file_name_test} of length: {len(final_data_test)}")
end = time.time()
print(f"Took: {end - start:.2f} seconds.")



