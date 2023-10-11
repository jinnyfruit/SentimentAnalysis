import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import pandas as pd

fold_num = 0

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('facebook/muppet-roberta-large')

# Load the trained model
model_path = f'best_model_{fold_num}.pt'
model = AutoModelForSequenceClassification.from_pretrained('facebook/muppet-roberta-large', num_labels=3)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Load test data
test_data = pd.read_csv('data/test.csv')  # Replace 'test_file_path.csv' with your test file path

# Create a custom dataset for test data
class TestDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', max_length=256, return_tensors='pt', truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

test_dataset = TestDataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inference
all_preds = []
all_ids = []

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

all_ids = test_data['id'].tolist()

# Create submission file
submission = pd.DataFrame({
    'id': all_ids,
    'sentiment': all_preds
})

submission.to_csv('submission.csv', index=False)

print("Inference complete and submission file saved!")
