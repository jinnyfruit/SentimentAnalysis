import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Load your train.csv dataset
data = pd.read_csv('data/train.csv')

# Define the number of folds
n_splits = 5

# Initialize the StratifiedKFold object
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create a directory to save the split datasets
import os
os.makedirs('splits', exist_ok=True)

# Create indices for the folds
fold_indices = list(skf.split(data['text'], data['sentiment']))

# Iterate through the folds
for fold, (train_idx, val_idx) in enumerate(fold_indices):
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]

    # Save the train and validation datasets
    train_data.to_csv(f'splits/train_fold_{fold}.csv', index=False)
    val_data.to_csv(f'splits/validation_fold_{fold}.csv', index=False)

    print(f"Fold {fold + 1}: Train samples = {len(train_data)}, Validation samples = {len(val_data)}")
