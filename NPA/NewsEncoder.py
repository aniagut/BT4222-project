import torch
from torch import nn


class ArticleEncoder(nn.Module):
    def __init__(self, embedding_dim, additional_features_dim, num_filters, kernel_size):
        super(ArticleEncoder, self).__init__()
        self.conv = nn.Conv1d(embedding_dim + additional_features_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, embeddings, additional_features):
        # Assume additional_features are concatenated to embeddings along the feature dimension
        combined_features = torch.cat((embeddings, additional_features), dim=2)
        combined_features = combined_features.permute(0, 2, 1)  # Adjust shape for Conv1d
        conv_out = self.conv(combined_features)
        pooled_out = self.pool(self.relu(conv_out))
        return pooled_out.squeeze(-1)




