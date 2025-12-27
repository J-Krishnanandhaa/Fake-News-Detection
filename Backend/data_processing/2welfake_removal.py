import pandas as pd

# File paths
welfake_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\welfake\\WELFake_Dataset.csv'
kaggle_fake_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\Fake.csv'
kaggle_true_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\True.csv'
output_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\welfake\\WELFake_Dataset_filtered.csv'

# Combine helper
def combine_text(title, text):
    return (str(title) + " " + str(text)).strip()

# Load datasets
df_wf = pd.read_csv(welfake_path).iloc[:, 1:]
df_kf = pd.read_csv(kaggle_fake_path)
df_kt = pd.read_csv(kaggle_true_path)

# Create combined text columns for comparison
df_wf['text'] = df_wf.apply(lambda x: combine_text(x['title'], x['text']), axis=1)
df_kf['text'] = df_kf.apply(lambda x: combine_text(x['title'], x['text']), axis=1)
df_kt['text'] = df_kt.apply(lambda x: combine_text(x['title'], x['text']), axis=1)

# Create sets of texts to compare
kaggle_texts = set(df_kf['text']) | set(df_kt['text'])

# Filter: Keep only rows NOT in kaggle datasets
df_filtered = df_wf[~df_wf['text'].isin(kaggle_texts)].copy()

# Drop the newly added 'text' column before saving (optional)
df_filtered.drop(columns=['text'], inplace=True)

# Save new dataset
df_filtered.to_csv(output_path, index=False)

print(f"✅ Filtered WELFake dataset saved at:\n{output_path}")
print(f"🧾 Original rows: {len(df_wf)}")
print(f"🧾 Filtered rows: {len(df_filtered)}")
print(f"🗑️  Removed rows: {len(df_wf) - len(df_filtered)}")
