import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
test_df = pd.read_csv('d:/fakeeee/backend/data/final_dataset/test.csv')
test_df['cleaned_text'] = test_df['cleaned_text'].fillna('').astype(str)

# Load model - specify the exact checkpoint path
checkpoint_path = 'd:/fakeeee/backend/src/newmodel/checkpoint-14628'
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Prepare test dataset
test_dataset = Dataset.from_pandas(test_df[['cleaned_text', 'encoded_labels']])

def tokenize_function(examples):
    return tokenizer(examples['cleaned_text'], truncation=True, padding='max_length', max_length=256)

test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.rename_column("encoded_labels", "labels")
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0)
    }

# Initialize trainer with metrics
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics
)

# Evaluate
results = trainer.evaluate(test_dataset)
print("\nTest set evaluation results:")
for metric, value in results.items():
    if metric != 'eval_loss':
        print(f"{metric}: {value:.4f}")