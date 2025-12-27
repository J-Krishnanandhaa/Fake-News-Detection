import pandas as pd

# Paths
paths = {
    'WELFake': r'd:\\fakeeee\\backend\\data\\raw_datasets\\welfake\\WELFake_Dataset_filtered.csv',
    'Kaggle_Fake': r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\Fake.csv',
    'Kaggle_True': r'd:\\fakeeee\\backend\\data\\raw_datasets\\kagglefakenews\\True.csv',
    'Indian': r'd:\\fakeeee\\backend\\data\\raw_datasets\\indianset\\news_dataset1.csv',
    'Kaggle2': r'd:\\fakeeee\\backend\\data\\raw_datasets\\kaggle2\\fake_and_real_news.csv',
    'Constraint_Train': r'd:\\fakeeee\\backend\\data\\raw_datasets\\covidset\\Constraint_Train.csv',
    'Constraint_Val': r'd:\\fakeeee\\backend\\data\\raw_datasets\\covidset\\Constraint_Val.csv',
    'BuzzFeed_Fake': r'd:\\fakeeee\\backend\\data\\raw_datasets\\buzzfeed\\BuzzFeed_fake_news_content.csv',
    'BuzzFeed_Real': r'd:\\fakeeee\\backend\\data\\raw_datasets\\buzzfeed\\BuzzFeed_real_news_content.csv',
}

# Load and print column names
print("🧾 Dataset Columns:\n")
for name, path in paths.items():
    try:
        df = pd.read_csv(path, encoding='ISO-8859-1' if 'Indian' in name else 'utf-8')
        print(f"{name:<20} ➤ {list(df.columns)}")
    except Exception as e:
        print(f"{name:<20} ➤ ❌ Error loading: {e}")
