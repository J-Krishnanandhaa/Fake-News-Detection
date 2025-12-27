import os
import pandas as pd

# Path to the main directory containing 6 dataset folders
base_path = r'd:\fakeeee\backend\data\raw_datasets'

# List all subdirectories (each should be a dataset folder)
dataset_folders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

# Loop through each dataset folder
for folder in dataset_folders:
    print(f"\n\n=== Dataset Folder: {os.path.basename(folder)} ===")

    # List all files in the dataset folder
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        if file.lower().endswith('.csv'):
            print(f"\n--- File: {file} ---")
            try:
                df = pd.read_csv(file_path, encoding='utf-8', engine='python')
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print("Sample Data:")
                print(df.head(3))
            except Exception as e:
                print(f"Error reading {file}: {e}")
        else:
            print(f"Skipping non-CSV file: {file}")
