import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from pathlib import Path

def generate_dataset_report():
    # Load combined dataset
    dataset_path = Path('d:/fakeeee/backend/data/raw_datasets/combined_dataset.csv')
    df = pd.read_csv(dataset_path)
    
    # Create report directory
    report_dir = Path('d:/fakeeee/backend/data/initial_reports')
    report_dir.mkdir(exist_ok=True)
    
    # 1. Basic Statistics - Modified to work with numeric labels
    stats = {
        'Total Samples': len(df),
        'Fake News Count': len(df[df['label'] == 0]),  # Changed to numeric comparison
        'True News Count': len(df[df['label'] == 1]),   # Changed to numeric comparison
        'Average Text Length': round(df['text'].str.len().mean(), 2),
        'Median Text Length': df['text'].str.len().median(),
        'Label Distribution': df['label'].value_counts(normalize=True).to_dict()  # Removed .str
    }
    
    # 2. Generate Visualizations
    plt.figure(figsize=(18, 12))
    
    # 2.1 Label Distribution
    plt.subplot(2, 2, 1)
    df['label'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
    plt.title('News Authenticity Distribution')
    
    # 2.2 Text Length Analysis
    plt.subplot(2, 2, 2)
    df['text_length'] = df['text'].str.len()
    sns.boxplot(data=df, x='label', y='text_length')
    plt.title('Text Length by Label')
    plt.ylabel('Characters')
    
    # 2.3 Word Clouds
    for i, label_num in enumerate([0, 1]):  # 0=fake, 1=real
        plt.subplot(2, 2, 3+i)
        label_data = df[df['label'] == label_num]['text'].dropna()  # Drop NaN values
        sample_size = min(1000, len(label_data))
        text = " ".join(label_data.sample(sample_size, random_state=42).astype(str)) if sample_size > 0 else ""
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"{'Fake' if label_num == 0 else 'Real'} News Word Cloud")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(report_dir/'dataset_visualizations.png', dpi=300, bbox_inches='tight')
    
    # 3. Save Statistics
    with open(report_dir/'dataset_statistics.txt', 'w') as f:
        f.write("DATASET ANALYSIS REPORT\n======================\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Report successfully generated in: {report_dir}")

if __name__ == "__main__":
    generate_dataset_report()

