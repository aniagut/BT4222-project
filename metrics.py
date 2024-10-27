import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# AUC
def calculate_auc(data):
    if data['clicked'].nunique() < 2:  # AUC needs at least one positive and one negative label
        return np.nan
    return roc_auc_score(data['clicked'], data['predicted_score'])

# MRR
def calculate_mrr(data):
    sorted_data = data.sort_values(by='predicted_score', ascending=False)
    ranks = sorted_data['clicked'].values
    for rank, click in enumerate(ranks, start=1):
        if click == 1:
            return 1 / rank
    return 0

# NDCG
def dcg(scores, k):
    return sum([score / np.log2(idx + 2) for idx, score in enumerate(scores[:k])])

def calculate_ndcg(data, k):
    sorted_data = data.sort_values(by='predicted_score', ascending=False)
    ideal_sorted_data = data.sort_values(by='clicked', ascending=False)
    dcg_k = dcg(sorted_data['clicked'].values, k)
    idcg_k = dcg(ideal_sorted_data['clicked'].values, k)
    return dcg_k / idcg_k if idcg_k > 0 else 0