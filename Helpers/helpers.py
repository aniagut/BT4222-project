from sklearn.model_selection import train_test_split


def split_by_session_train_val_test(data, test_size=0.2, val_size=0.1, random_state=42):
	# Get unique sessions
	unique_sessions = data['session_id'].unique()

	# Split sessions into train and remaining (validation + test)
	train_sessions, remaining_sessions = train_test_split(unique_sessions,
	                                                      test_size=test_size + val_size,
	                                                      random_state=random_state)

	# Calculate the proportion of remaining sessions to allocate to test
	proportion_test = test_size / (test_size + val_size)

	# Split remaining sessions into validation and test
	val_sessions, test_sessions = train_test_split(remaining_sessions,
	                                               test_size=proportion_test,
	                                               random_state=random_state)

	# Create train, validation, and test sets
	train_data = data[data['session_id'].isin(train_sessions)]
	val_data = data[data['session_id'].isin(val_sessions)]
	test_data = data[data['session_id'].isin(test_sessions)]

	return train_data, val_data, test_data


def get_historical_clicks(row, all_clicks, max_lookback=pd.Timedelta(days=2)):
    # Filter clicks prior to the current impression within the lookback window
    user_clicks = all_clicks[(all_clicks['user_id'] == row['user_id']) &
                             (all_clicks['impression_time'] < row['impression_time']) &
                             (all_clicks['impression_time'] >= row['impression_time'] - max_lookback)]
    return user_clicks['article_ids_clicked'].tolist()  # Return a list of article IDs

def get_historical_not_clicked(row, all_clicks, max_lookback=pd.Timedelta(days=2)):
    user_clicks =  all_clicks[(all_clicks['user_id'] == row['user_id'])
    & (all_clicks['impression_time'] < row['impression_time'])
    & (all_clicks['impression_time'] >= row['impression_time'] - max_lookback)]
    return user_clicks['article_ids_not_clicked'].unique().tolist()

""""
# Apply the function to each row to get lists of IDs
data['historical_article_ids'] = data.apply(get_historical_clicks, axis=1, all_clicks=data)
data['historical_article_ids_not_clicked'] = data.apply(get_historical_not_clicked,axis=1, all_clicks=data)
"""


import random

def get_historical_for_user_per_impression(row, negative_samples=4, max_lookback=pd.Timedelta(days=2)):
  # select randomly 4 negative samples so from the
  positive_samples_count = len(row['historical_article_ids'])
  negative_samples = random.sample(row['historical_article_ids_not_clicked'], negative_samples * positive_samples_count)
  return row['historical_article_ids'],  negative_samples


def get_embeddings_for(row, embeddings):
  selected_art_pos, selected_art_neg = get_historical_for_user_per_impression(row, negative_samples=4)
  return embeddings[embeddings['article_id'].isin(selected_art_pos)], embeddings[embeddings['article_id'].isin(selected_art_neg)]


def calculate_click_percentage(row):
    if not row['article_ids_inview']:
        return 0
    total_articles = len(row['article_ids_inview'].split(','))
    clicked_articles = len(row['article_ids_clicked'].split(','))
    if total_articles > 0:
        return clicked_articles / total_articles
    else:
        return 0



import pandas as pd
def add_historical_embeddings_user_impression(data, embeddings, max_lookback=pd.Timedelta(days=2),
                                              nof_negative_samples_ratio=4):
	# Initialize list to collect data rows
	embeddings_user_impression = []

	# Group data by 'user_id' and 'impression_time'
	grouped_data = data.groupby(['user_id', 'impression_time'])

	for (user_id, impression_time), group in grouped_data:
		# Get the indices of articles clicked and not clicked
		positive_samples_idx = [item for sublist in group['historical_article_ids'].dropna().tolist() for item in
		                        sublist]
		negative_samples_possible = [item for sublist in group['historical_article_ids_not_clicked'].dropna().tolist()
		                             for item in sublist]

		# Ensure there are enough negative samples
		nof_negative_samples = min(len(negative_samples_possible),
		                           nof_negative_samples_ratio * len(positive_samples_idx))
		negative_samples_idx = random.sample(negative_samples_possible,
		                                     nof_negative_samples) if negative_samples_possible else []

		# Fetch embeddings
		pos_embeddings = embeddings[embeddings['article_id'].isin(positive_samples_idx)]
		neg_embeddings = embeddings[embeddings['article_id'].isin(negative_samples_idx)]

		# Append new row to the list
		embeddings_user_impression.append({
			'user_id': user_id,
			'impression_time': impression_time,
			'pos_embeddings': pos_embeddings,
			'neg_embeddings': neg_embeddings
		})

	# Convert list to DataFrame
	return pd.DataFrame(embeddings_user_impression)