import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import pandas as pd

fold_num = 0

# Load data
train_data = pd.read_csv(f'splits/train_fold_{fold_num}.csv')
val_data = pd.read_csv(f'splits/validation_fold_{fold_num}.csv')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('facebook/muppet-roberta-large')
model = AutoModelForSequenceClassification.from_pretrained('facebook/muppet-roberta-large', num_labels=3)  # Assuming 3 labels for sentiment

# Create a custom dataset
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        sentiment = self.data.iloc[idx]['sentiment']
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=256, return_tensors='pt', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(sentiment, dtype=torch.long)
        }

train_dataset = SentimentDataset(train_data, tokenizer)
val_dataset = SentimentDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop with early stopping based on Macro F1 Score
best_f1 = 0
patience = 3
no_improve = 0

for epoch in range(10):  # Assuming 10 epochs, can be changed
    model.train()
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch {epoch + 1}, Macro F1 Score: {f1:.4f}")

    # Early stopping
    if f1 > best_f1:
        best_f1 = f1
        no_improve = 0
        torch.save(model.state_dict(), f'best_model_{fold_num}.pt')
    else:
        no_improve += 1
        if no_improve == patience:
            print("Early stopping!")
            break

print("Training complete!")
