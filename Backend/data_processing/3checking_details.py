import pandas as pd

# Paths
welfake_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\welfake\\WELFake_Dataset_filtered.csv'
kaggle_fake_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\Fake.csv'
kaggle_true_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\True.csv'
indian_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\indianset\\news_dataset1.csv'
kaggle2_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\kaggle2\\fake_and_real_news.csv'
constraint_train_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\covidset\\Constraint_Train.csv'
constraint_val_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\covidset\\Constraint_Val.csv'
buzz_fake_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\buzzfeed\\BuzzFeed_fake_news_content.csv'
buzz_real_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\buzzfeed\\BuzzFeed_real_news_content.csv'

# Normalize function to standardize labels
def normalize_label(label):
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    if label in ['0', 'fake', 'false', 'f']:
        return 'fake'
    elif label in ['1', 'real', 'true', 't']:
        return 'real'
    else:
        return None  # Unknown label

# Storage for label counts
label_counts = {}

# Load and normalize WELFake
df_wf = pd.read_csv(welfake_path)
df_wf['label_norm'] = df_wf['label'].apply(normalize_label)
label_counts['WELFake'] = df_wf['label_norm'].value_counts().to_dict()

# Load Kaggle Fake and True datasets (all fake or all real)
df_kf = pd.read_csv(kaggle_fake_path)
label_counts['Kaggle_Fake'] = {'fake': len(df_kf)}

df_kt = pd.read_csv(kaggle_true_path)
label_counts['Kaggle_True'] = {'real': len(df_kt)}

# Load Indian dataset and normalize labels if column exists
df_ind = pd.read_csv(indian_path, encoding='ISO-8859-1')
if 'label' in df_ind.columns:
    df_ind['label_norm'] = df_ind['label'].apply(normalize_label)
    label_counts['Indian'] = df_ind['label_norm'].value_counts().to_dict()
else:
    label_counts['Indian'] = {'unknown': len(df_ind)}  # fallback

# Load Kaggle2 dataset and normalize labels
df_k2 = pd.read_csv(kaggle2_path)
df_k2['label_norm'] = df_k2['label'].apply(normalize_label)
label_counts['Kaggle2'] = df_k2['label_norm'].value_counts().to_dict()

# Load Constraint COVID datasets and normalize labels
df_c_train = pd.read_csv(constraint_train_path)
df_c_val = pd.read_csv(constraint_val_path)
combined_c = pd.concat([df_c_train, df_c_val])
combined_c['label_norm'] = combined_c['label'].apply(normalize_label)
label_counts['Constraint_COVID'] = combined_c['label_norm'].value_counts().to_dict()

# Load BuzzFeed datasets (all fake or all real)
df_buzz_fake = pd.read_csv(buzz_fake_path)
label_counts['BuzzFeed_Fake'] = {'fake': len(df_buzz_fake)}

df_buzz_real = pd.read_csv(buzz_real_path)
label_counts['BuzzFeed_Real'] = {'real': len(df_buzz_real)}

# --------------------------
# Final Summary
# --------------------------
print("\n📊 Label Counts Per Dataset:\n")
total_real, total_fake = 0, 0

for dataset, counts in label_counts.items():
    real = counts.get('real', 0)
    fake = counts.get('fake', 0)
    total_real += real
    total_fake += fake
    print(f"{dataset:<20} => real: {real:5d}, fake: {fake:5d}")

print("\n🧾 TOTAL")
print(f"Total REAL: {total_real}")
print(f"Total FAKE: {total_fake}")
print(f"Total ALL : {total_real + total_fake}")
