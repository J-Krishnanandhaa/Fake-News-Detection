import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_split_report():
    data_dir = Path('d:/fakeeee/backend/data/final_dataset')
    train_path = data_dir / 'train.csv'
    val_path = data_dir / 'val.csv'
    test_path = data_dir / 'test.csv'

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    sizes = [len(train_df), len(val_df), len(test_df)]
    labels = ['Train', 'Validation', 'Test']

    def text_length_stats(df):
        lengths = df['cleaned_text'].astype(str).str.len()
        return {
            'avg_length': round(lengths.mean(), 2),
            'median_length': lengths.median()
        }

    train_stats = text_length_stats(train_df)
    val_stats = text_length_stats(val_df)
    test_stats = text_length_stats(test_df)

    report_dir = Path('d:/fakeeee/backend/data/final_report')
    report_dir.mkdir(exist_ok=True)

    # Pie chart for split proportions
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#99ff99', '#ffcc99'])
    plt.title('Dataset Split Proportions')
    plt.savefig(report_dir / 'split_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Pie charts for fake/real in each split
    for df, label in zip([train_df, val_df, test_df], labels):
        counts = df['encoded_labels'].value_counts().sort_index()
        # Assuming 0: fake, 1: real
        pie_labels = ['Fake', 'Real']
        pie_sizes = [counts.get(0, 0), counts.get(1, 0)]
        plt.figure(figsize=(5, 5))
        plt.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
        plt.title(f'{label} Set: Fake vs Real')
        plt.savefig(report_dir / f'{label.lower()}_fake_real_pie.png', dpi=300, bbox_inches='tight')
        plt.close()

    stats_path = report_dir / 'split_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("DATASET SPLIT REPORT\n===================\n\n")
        for label, size, stats, df in zip(
            labels, sizes, [train_stats, val_stats, test_stats], [train_df, val_df, test_df]
        ):
            fake_count = (df['encoded_labels'] == 0).sum()
            real_count = (df['encoded_labels'] == 1).sum()
            f.write(f"{label} Set Size: {size}\n")
            f.write(f"  Fake Count: {fake_count}\n")
            f.write(f"  Real Count: {real_count}\n")
            f.write(f"  Average Text Length: {stats['avg_length']}\n")
            f.write(f"  Median Text Length: {stats['median_length']}\n")
        f.write(f"\nTotal Samples: {sum(sizes)}\n")
        f.write(f"Proportions: {dict(zip(labels, [round(s/sum(sizes), 3) for s in sizes]))}\n")

    print(f"Split report and pie charts generated in: {report_dir}")

if __name__ == "__main__":
    generate_split_report()