import pandas as pd

# Output file
output_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\combined_dataset.csv'

# Combine helper
def combine_text(title, text):
    return (str(title) + " " + str(text)).strip()

combined_data = []

# --- WELFake ---
df = pd.read_csv(r'd:\\fakeeee\\backend\\data\\raw_datasets\\welfake\\WELFake_Dataset_filtered.csv')
df['text'] = df['title'].astype(str)
df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
combined_data.append(df[['text', 'label']])

# --- Kaggle Fake ---
df = pd.read_csv(r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\Fake.csv')
df['text'] = df.apply(lambda x: combine_text(x['title'], x['text']), axis=1)
df['label'] = 0
combined_data.append(df[['text', 'label']])

# --- Kaggle True ---
df = pd.read_csv(r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\True.csv')
df['text'] = df.apply(lambda x: combine_text(x['title'], x['text']), axis=1)
df['label'] = 1
combined_data.append(df[['text', 'label']])

# --- Indian ---
df = pd.read_csv(r'd:\\fakeeee\\backend\\data\\raw_datasets\\indianset\\news_dataset1.csv', encoding='ISO-8859-1')
df['label'] = df['label'].str.upper().map({'FAKE': 0, 'REAL': 1})
df['text'] = df['text'].astype(str)
combined_data.append(df[['text', 'label']])

# --- Kaggle2 ---
df = pd.read_csv(r'd:\\fakeeee\\backend\\data\\raw_datasets\\kaggle2\\fake_and_real_news.csv')
df['label'] = df['label'].astype(str).str.upper().map({'FAKE': 0, 'REAL': 1})
df['text'] = df['Text'].astype(str)
combined_data.append(df[['text', 'label']])

# --- Constraint COVID ---
train_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\covidset\\Constraint_Train.csv'
val_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\covidset\\Constraint_Val.csv'
df = pd.concat([
    pd.read_csv(train_path),
    pd.read_csv(val_path)
])
df['label'] = df['label'].str.lower().map({'fake': 0, 'real': 1})
df['text'] = df['tweet'].astype(str)
combined_data.append(df[['text', 'label']])

# --- BuzzFeed Fake ---
df = pd.read_csv(r'd:\\fakeeee\\backend\\data\\raw_datasets\\buzzfeed\\BuzzFeed_fake_news_content.csv')
df['text'] = df.apply(lambda x: combine_text(x['title'], x['text']), axis=1)
df['label'] = 0
combined_data.append(df[['text', 'label']])

# --- BuzzFeed Real ---
df = pd.read_csv(r'd:\\fakeeee\\backend\\data\\raw_datasets\\buzzfeed\\BuzzFeed_real_news_content.csv')
df['text'] = df.apply(lambda x: combine_text(x['title'], x['text']), axis=1)
df['label'] = 1
combined_data.append(df[['text', 'label']])

# --- Combine All ---
final_df = pd.concat(combined_data, ignore_index=True)
final_df.dropna(subset=['text', 'label'], inplace=True)
final_df.drop_duplicates(subset=['text', 'label'], inplace=True)
final_df = final_df.sample(frac=1).reset_index(drop=True)

# --- Save ---
final_df.to_csv(output_path, index=False)
print(f"✅ Combined dataset saved to: {output_path}")
print(f"📝 Total samples: {len(final_df)}")
