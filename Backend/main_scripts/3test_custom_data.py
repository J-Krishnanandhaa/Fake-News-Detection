import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load trained model
checkpoint_path = 'd:/fakeeee/backend/src/newmodel/checkpoint-14628'
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model.eval()  # Set model to evaluation mode

# Input custom text
custom_data = [
    "Breaking news: The government confirmed an alien sighting in NYC.",
    "pojk"
]

# Clean invalid entries
custom_data = [t for t in custom_data if isinstance(t, str) and t.strip()]
if not custom_data:
    raise ValueError("No valid text provided for prediction.")

# Prediction function
def predict(texts):
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()

# Get predictions
predictions = predict(custom_data)

# Display results
for text, prob in zip(custom_data, predictions):
    print(f"\n📰 Text: {text[:200]}...")
    print(f"\n🧪 Fake probability: {prob[0]:.4f}")
    print(f"🧪 Real probability: {prob[1]:.4f}")
    predicted_class = prob.argmax()
    print(f"\n📌 Prediction: {'FAKE' if predicted_class == 0 else 'REAL'}")
