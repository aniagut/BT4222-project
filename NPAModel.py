import torch
from torch import nn
from NewsEncoder import NewsEncoder
from UserAttention import UserAttention


import torch
from torch import nn

class NPAModel(nn.Module):
    def __init__(self, num_words, embedding_dim, num_filters, kernel_size, num_user_features, num_article_stats):
        super(NPAModel, self).__init__()
        self.news_encoder = NewsEncoder(num_words, embedding_dim, num_filters, kernel_size)
        self.user_attention = UserAttention(num_user_features, num_filters)
        # Adjust input dimensions to include additional features
        self.fc1 = nn.Linear(num_filters + num_user_features + num_article_stats, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, news_input, user_features, article_stats):
        news_vector = self.news_encoder(news_input)
        user_news_vector = self.user_attention(user_features, news_vector)
        # Concatenate additional features
        combined_vector = torch.cat((user_news_vector, user_features, article_stats), dim=1)
        x = self.fc1(combined_vector)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
