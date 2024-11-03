import torch.nn as nn

class NewsEncoder(nn.Module):
    def __init__(self, num_words, embedding_dim, num_filters, kernel_size):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding(num_words, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.squeeze(-1)



