from torch import nn
import torch.nn.functional as F
import torch
"""
Pairwise ranking losses compare pairs of items within
the same session and compute gradients
that push the score of the relevant item higher
than the score of the less relevant item.
"""


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, targets, mask):
        # Filter scores and targets according to the mask
        # Only consider non-padded data for calculating the loss
        filtered_scores = scores[mask]
        filtered_targets = targets[mask]

        positive_indices = filtered_targets > 0
        negative_indices = filtered_targets <= 0

        if positive_indices.any() and negative_indices.any():
            positive_scores = filtered_scores[positive_indices]
            negative_scores = filtered_scores[negative_indices]

            # Compute pairwise differences within the session
            diff = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
            loss = F.relu(self.margin - diff).mean()  # Applying hinge loss
            return loss
        else:
            # Return a zero loss if no valid pairs are available
            return torch.tensor(0.0, device=scores.device, requires_grad=True)


