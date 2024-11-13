import torch.nn as nn
import torch


"""
How the class works together?

User Features Processing: 

UserFeatureEncoder takes in both static and dynamic features and 
produces a single vector that represents the user's overall 
profile at a given moment.

Attention Mechanism Application:

The output from UserFeatureEncoder is then fed into UserAttention,
which uses this user profile to modulate the importance of
different articles. This step is crucial for tailoring the model's output
to the individual user, emphasizing articles predicted 
to be most relevant based on the user's combined long-term preferences
and immediate context.
"""
class UserAttention(nn.Module):
    """
    Purpose:
    The UserAttention module is designed to adjust
    the importance of different news article representations based on
    the user's comprehensive profile.
    It uses the combined user features to determine
    how relevant each piece of news is to the user,
    potentially enhancing personalization by focusing on articles, that align
    more closely with the user's interests and current context.

    How it works:
    It transforms the combined user features (from UserFeatureEncoder)
    to align with the news feature space, computes attention scores
    between the user's features and each news article representation,
    and then applies these attention scores to prioritize
    or deprioritize certain news articles based on the user's preferences.
    """
    def __init__(self, user_feature_size, news_feature_size):
        super(UserAttention, self).__init__()
        self.user_to_news_transform = nn.Linear(user_feature_size, news_feature_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, news_rep, user_features):
        # Transform user features to match news features dimension
        transformed_user_features = self.user_to_news_transform(user_features)
        # Calculate attention scores
        attention_scores = torch.bmm(news_rep, transformed_user_features.unsqueeze(2))
        attention_weights = self.softmax(attention_scores.squeeze(2))
        # Apply attention weights
        attended_news = attention_weights.unsqueeze(2) * news_rep
        return attended_news.sum(1)



class UserFeatureEncoder(nn.Module):
    """
    Purpose:
    This module's primary function is to process and combine
    user-specific features—both dynamic and static—into a
    unified vector that can be used for further processing.

    How it works:
    It takes dynamic features which might vary from session to session,
    like the device type and static features, which remain constant per user,
    such as subscription status or general preferences
    and processes each with separate linear layers.
    The outputs of these layers are then combined (concat)
    to form a comprehensive representation of the user's profile.
    """
    def __init__(self, dynamic_feature_dim, static_feature_dim, output_dim):
        super(UserFeatureEncoder, self).__init__()
        self.dynamic_processor = nn.Linear(dynamic_feature_dim, output_dim)
        self.static_processor = nn.Linear(static_feature_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, dynamic_features, static_features):
        dynamic_out = self.relu(self.dynamic_processor(dynamic_features))
        static_out = self.relu(self.static_processor(static_features))
        combined_features = torch.cat((dynamic_out, static_out), dim=1)
        return combined_features
