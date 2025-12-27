import pandas as pd
import re
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words).strip()

def preprocess_dataset(input_path, model_name='bert-base-uncased'):
    df = pd.read_csv(input_path)

    # Drop rows with nulls in text or label
    df = df.dropna(subset=['text', 'label'])

    # Apply cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df['tokenized'] = df['cleaned_text'].apply(
        lambda x: tokenizer.encode(x, truncation=True, max_length=512)
    )

    # Encode labels
    label_encoder = LabelEncoder()
    df['encoded_labels'] = label_encoder.fit_transform(df['label'])

    # Split into train, val, and test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['encoded_labels'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['encoded_labels'])

    return train_df, val_df, test_df, tokenizer, label_encoder

if __name__ == "__main__":
    input_path = Path('d:/fakeeee/backend/data/raw_datasets/combined_dataset.csv')
    output_dir = Path('d:/fakeeee/backend/data/final_dataset')
    output_dir.mkdir(exist_ok=True)

    train_df, val_df, test_df, tokenizer, _ = preprocess_dataset(input_path)

    # Save only cleaned and encoded versions
    train_df[['cleaned_text', 'encoded_labels']].to_csv(output_dir / 'train.csv', index=False)
    val_df[['cleaned_text', 'encoded_labels']].to_csv(output_dir / 'val.csv', index=False)
    test_df[['cleaned_text', 'encoded_labels']].to_csv(output_dir / 'test.csv', index=False)

    print(f"✅ Processed datasets saved to: {output_dir}")
