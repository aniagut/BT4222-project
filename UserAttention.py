import torch.nn as nn
import torch

class UserAttention(nn.Module):
    def __init__(self, user_features_dim, news_features_dim):
        super(UserAttention, self).__init__()
        self.user_projection = nn.Linear(user_features_dim, news_features_dim)

    def forward(self, news_embeddings, user_features):
        # Expand and project user features to match news features dimensions
        user_features_projected = self.user_projection(user_features).unsqueeze(1)
        scores = torch.bmm(user_features_projected, news_embeddings.unsqueeze(-1)).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=1)
        return (attention_weights.unsqueeze(-1) * news_embeddings).sum(dim=1)