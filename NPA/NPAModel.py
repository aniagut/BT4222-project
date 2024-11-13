import torch
from torch import nn
from NewsEncoder import ArticleEncoder
from UserAttention import UserAttention, UserFeatureEncoder


import torch
from torch import nn

class NPAModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_filters,
                 kernel_size,
                 dynamic_feature_dim,
                 static_feature_dim,
                 additional_article_features_dim
                 ):

        super(NPAModel, self).__init__()
        # News article encoder handles article embeddings
        # and additional article-specific features.
        self.article_encoder = ArticleEncoder(
            embedding_dim,
            num_filters,
            kernel_size,
            additional_article_features_dim
        )

        # User feature encoder processes both static
        # and dynamic user features into a unified vector.
        self.user_feature_encoder = UserFeatureEncoder(
            dynamic_feature_dim,
            static_feature_dim,
            num_filters
        )

        # User attention applies attention mechanism based
        # on user features to news representations.
        self.user_attention = UserAttention(num_filters, num_filters)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self,
                article_embeddings,
                article_features,
                dynamic_features,
                static_features
                ):
        news_rep = self.article_encoder(article_embeddings, article_features)
        user_rep = self.user_feature_encoder(dynamic_features, static_features)
        attended_news_rep = self.user_attention(news_rep, user_rep)
        out = self.fc(attended_news_rep)
        return torch.sigmoid(out)

