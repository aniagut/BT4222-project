from torch.utils.data import Dataset
import torch

class ArticlesDataset(Dataset):
    def __init__(self, articles, tokenizer):
        self.articles = articles
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles.iloc[idx]  # Access the article
        article_text = article['body']

        # Tokenize the text
        inputs = self.tokenizer(article_text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

        # Extract input_ids and attention_mask from the tokenized data
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'article_id': torch.tensor(article['article_id'], dtype=torch.long),
            'category': torch.tensor(article['category'], dtype=torch.long),
            'sentiment_score': torch.tensor(article['sentiment_score'], dtype=torch.float),
            'premium': torch.tensor(article['premium'], dtype=torch.bool)
        }

