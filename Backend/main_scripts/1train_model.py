import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from datasets import Dataset
import matplotlib.pyplot as plt
import os

# Device check
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

# Load and preprocess data
train_df = pd.read_csv('d:/fakeeee/backend/data/final_dataset/train.csv')
val_df = pd.read_csv('d:/fakeeee/backend/data/final_dataset/val.csv')

for df in [train_df, val_df]:
    df['cleaned_text'] = df['cleaned_text'].fillna('').astype(str)

# Initialize DistilBERT model and tokenizer
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples['cleaned_text'], truncation=True, padding='max_length', max_length=256)

# Prepare datasets
train_dataset = Dataset.from_pandas(train_df[['cleaned_text', 'encoded_labels']])
val_dataset = Dataset.from_pandas(val_df[['cleaned_text', 'encoded_labels']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("encoded_labels", "labels")
val_dataset = val_dataset.rename_column("encoded_labels", "labels")
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Metrics calculation
# Create directory for saving plots
os.makedirs('d:/fakeeee/backend/src/newmodel/plots', exist_ok=True)

# Modified compute_metrics to include basic plotting with values
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0)
    }
    
    # Basic plot of metrics
    plt.figure(figsize=(10, 5))
    bars = plt.bar(metrics.keys(), metrics.values())
    plt.title(f'Metrics at Epoch {trainer.state.epoch}')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Annotate bars with metric values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')
    
    plt.savefig(f'd:/fakeeee/backend/src/newmodel/plots/metrics_epoch_{trainer.state.epoch}.png')
    plt.close()
    
    return metrics

# Training arguments
training_args = TrainingArguments(
    output_dir='d:/fakeeee/backend/src/newmodel',
    num_train_epochs=6,  
    per_device_train_batch_size=24,
    per_device_eval_batch_size=24,
    fp16=True,
    gradient_accumulation_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='d:/fakeeee/backend/src/newmodel',
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    save_total_limit=2,
    report_to="none",
    logging_strategy="steps",
    logging_first_step=True,
    logging_steps=1
)

# Initialize and run trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
