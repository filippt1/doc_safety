import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

INPUT_DIR = 'results/confusion_matrices'  # path to confusion matrix CSV files
OUTPUT_DIR = 'results/plots'  # path to save the generated plots
TOP_N = 18  # number of top characters to display

os.makedirs(OUTPUT_DIR, exist_ok=True)


# generate and save heatmap plot with log scale
def plot_heatmap(df, title, filename):
    plt.figure(figsize=(14, 12))

    cm = pd.crosstab(df['Truth'], df['Pred'])

    top_rows = df['Truth'].value_counts().nlargest(TOP_N).index

    valid_cols = [c for c in top_rows if c in cm.columns]
    cm_filtered = cm.loc[top_rows, valid_cols]

    if cm_filtered.empty:
        print(f"Skipping empty plot: {filename}")
        plt.close()
        return

    sns.heatmap(cm_filtered,
                annot=True,
                fmt='d',
                cmap='Blues',
                norm=LogNorm(vmin=1, vmax=cm_filtered.max().max()),
                cbar_kws={'label': 'Count (Log Scale)'})

    plt.title(title, fontsize=16)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")


# process each confusion matrix CSV file
files = [f for f in os.listdir(INPUT_DIR) if f.startswith('confusion_') and f.endswith('.csv')]

print(f"Found {len(files)} confusion matrices. Generating plots with Log Scale...")

# loop through files and generate plots
for csv_file in files:
    try:
        base_name = os.path.splitext(csv_file)[0]
        parts = base_name.split('_')

        if len(parts) >= 3:
            model_name = parts[1]
            engine_name = parts[2]
        else:
            model_name = "Unknown"
            engine_name = base_name

        df = pd.read_csv(os.path.join(INPUT_DIR, csv_file))
        df.replace(' ', '<SPACE>', inplace=True)  # handle space character in labels
        if df.empty:
            continue

        plot_heatmap(df,
                     title=f'Confusion Matrix: {model_name} + {engine_name}',
                     filename=f'heatmap_{model_name}_{engine_name}.png')

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

print("All plots generated.")
