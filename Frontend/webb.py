''' streamlit run d:\fakeeee\frontend\webb.py '''
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Set up Streamlit page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Title
st.title("📰 Fake News Detection")
st.markdown("Enter one or more news headlines/articles separated by semicolons.")

# Load model and tokenizer
MODEL_PATH = 'd:/fakeeee/backend/src/newmodel/checkpoint-14628'
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Prediction function
def batch_predict(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()

# Heuristic to check unfamiliar input based on excessive token fragmentation
def is_unfamiliar(text, tokenizer, threshold=1.5):
    tokens = tokenizer.tokenize(text)
    word_count = len(text.split())
    token_count = len(tokens)
    if word_count == 0:
        return True
    return (token_count / word_count) > threshold

# --- Multi-text input ---
st.subheader("🔍 Predict from Multiple Texts")
text_input = st.text_area("📌 Input News Text(s): (use semicolons to separate)", height=150)

if st.button("Predict"):
    if text_input.strip():
        with st.spinner("Predicting..."):
            texts = [t.strip() for t in text_input.split(";") if t.strip()]
            predictions = batch_predict(texts)

            st.subheader("📊 Results")
            for i, (text, probs) in enumerate(zip(texts, predictions), start=1):
                fake_prob, real_prob = float(probs[0]), float(probs[1])
                prediction = "Fake" if fake_prob > real_prob else "Real"
                confidence = max(fake_prob, real_prob)
                label_color = "🟥" if prediction == "Fake" else "🟩"
                confidence_label = f"Confidence ({prediction})"

                st.markdown(f"**{i}.** _{text}_")
                st.write(f"Prediction: {label_color} **{prediction}**")
                st.progress(confidence, text=confidence_label)
                st.markdown(f"- **Fake Probability:** `{fake_prob:.4f}`")
                st.markdown(f"- **Real Probability:** `{real_prob:.4f}`")

                # Check and show warning for unfamiliar input
                if is_unfamiliar(text, tokenizer):
                    st.warning("⚠️ This input appears unfamiliar or very different from the training data. Prediction may be less reliable.")

                st.markdown("---")
    else:
        st.warning("⚠️ Please enter at least one news text.")
