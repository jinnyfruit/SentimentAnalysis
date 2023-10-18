import pandas as pd
from transformers import AutoTokenizer
import transformers
import torch
import csv

# Load the training data
df = pd.read_csv('splits/train_fold_0.csv')

# Filter data with sentiment values 0 and 1
filtered_data = df[df['sentiment'].isin([0, 1])]

# Initialize the Llama2 model and tokenizer
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Create a list to store the augmented data
augmented_data = []

# Generate paraphrases and store in dataaugmentation.csv
for index, row in filtered_data.iterrows():
    text = row['text']
    sequences = pipeline(
        text,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=100,
    )

    for seq in sequences:
        paraphrased_text = seq['generated_text']
        print(paraphrased_text)
        augmented_data.append([row['id'], text, row['sentiment'], paraphrased_text])

# Save the augmented data to dataaugmentation.csv
with open('dataaugmentation.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['id', 'text', 'sentiment', 'paraphrased_text'])
    csvwriter.writerows(augmented_data)

print("Data augmentation complete. Augmented data saved to dataaugmentation.csv.")
