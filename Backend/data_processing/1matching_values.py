import pandas as pd

# Paths
welfake_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\welfake\\WELFake_Dataset_filtered.csv'
kaggle_fake_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\Fake.csv'
kaggle_true_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\True.csv'
indian_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\indianset\\news_dataset1.csv'
kaggle2_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\kaggle2\\fake_and_real_news.csv'
buzz_fake_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\buzzfeed\\BuzzFeed_fake_news_content.csv'
buzz_real_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\buzzfeed\\BuzzFeed_real_news_content.csv'
covid_train_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\covidset\\Constraint_Train.csv'
covid_val_path = r'd:\\fakeeee\\backend\\data\\raw_datasets\\covidset\\Constraint_Val.csv'

# Combine helper
def combine_text(title, text):
    return (str(title) + " " + str(text)).strip()

# Load datasets
df_wf = pd.read_csv(welfake_path)  
if 'title' in df_wf.columns and 'text' not in df_wf.columns:
    df_wf['text'] = df_wf['title'].astype(str).str.strip()
df_wf['text'] = df_wf['text'].astype(str)

df_kf = pd.read_csv(kaggle_fake_path)
df_kf['text'] = df_kf.apply(lambda x: combine_text(x['title'], x['text']), axis=1)

df_kt = pd.read_csv(kaggle_true_path)
df_kt['text'] = df_kt.apply(lambda x: combine_text(x['title'], x['text']), axis=1)

df_ind = pd.read_csv(indian_path, encoding='ISO-8859-1')
df_ind['text'] = df_ind['text'].astype(str)

df_k2 = pd.read_csv(kaggle2_path)
df_k2.rename(columns={'Text': 'text'}, inplace=True)
df_k2['text'] = df_k2['text'].astype(str)

df_bf_fake = pd.read_csv(buzz_fake_path)
df_bf_fake['text'] = df_bf_fake.apply(lambda x: combine_text(x['title'], x['text']), axis=1)

df_bf_real = pd.read_csv(buzz_real_path)
df_bf_real['text'] = df_bf_real.apply(lambda x: combine_text(x['title'], x['text']), axis=1)

df_covid_train = pd.read_csv(covid_train_path)
df_covid_train.rename(columns={'tweet': 'text'}, inplace=True)
df_covid_train['text'] = df_covid_train['text'].astype(str)

df_covid_val = pd.read_csv(covid_val_path)
df_covid_val.rename(columns={'tweet': 'text'}, inplace=True)
df_covid_val['text'] = df_covid_val['text'].astype(str)

# Prepare list of datasets
datasets = {
    'WELFake': df_wf,
    'Kaggle_Fake': df_kf,
    'Kaggle_True': df_kt,
    'Indian': df_ind,
    'Kaggle2': df_k2,
    'BuzzFake': df_bf_fake,
    'BuzzReal': df_bf_real,
    'CovidTrain': df_covid_train,
    'CovidVal': df_covid_val
}

# Check pairwise matches
print("\n🔍 Matching texts between datasets:\n")
checked = set()
for name1, df1 in datasets.items():
    for name2, df2 in datasets.items():
        if name1 >= name2 or (name2, name1) in checked:
            continue
        checked.add((name1, name2))
        common_texts = set(df1['text']) & set(df2['text'])
        if common_texts:
            print(f"✅ {name1} <--> {name2}: {len(common_texts)} matching texts")




