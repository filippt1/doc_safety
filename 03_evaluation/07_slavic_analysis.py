import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

CZECH_CHARS = list("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")
MODELS = ['dncnn', 'drunet', 'pix2pix']
ENGINE = 'tesseract'

# Configuration
INPUT_DIR = 'results/confusion_matrices'
OUTPUT_DIR = 'results/plots_slavic'
TABLE_OUTPUT = f'results/slavic_stats_{ENGINE}.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_stats_table():
    results = []
    model_dfs = {}

    for model in MODELS:
        path = os.path.join(INPUT_DIR, f"confusion_{model}_{ENGINE}.csv")
        if os.path.exists(path):
            model_dfs[model] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found.")
            return

    all_truths = pd.concat([df['Truth'] for df in model_dfs.values()]).unique()
    present_czech_chars = sorted([c for c in CZECH_CHARS if c in all_truths])

    for char in present_czech_chars:
        row = {'character': char}
        for model in MODELS:
            df = model_dfs[model]
            subset = df[df['Truth'] == char]
            total = len(subset)

            if total > 0:
                correct = len(subset[subset['Pred'] == char])
                acc = correct / total

                errors = subset[subset['Pred'] != char]
                if not errors.empty:
                    most_common = errors['Pred'].mode()[0]
                    pct_common = len(errors[errors['Pred'] == most_common]) / total
                else:
                    most_common = "-"
                    pct_common = 0.0
            else:
                acc, most_common, pct_common = 0.0, "N/A", 0.0

            row['character occurrences'] = total
            row[f'{model} acc'] = f"{acc:.1%}"
            row[f'most common {model} error'] = most_common
            row[f'% most common error {model}'] = f"{pct_common:.1%}"

        results.append(row)

    df_table = pd.DataFrame(results)
    df_table = df_table.sort_values(by='character occurrences', ascending=False)
    # Reorder columns
    cols = ['character']
    for model in MODELS:
        cols.extend(['character occurrences', f'{model} acc', f'most common {model} error', f'% most common error {model}'])

    return df_table[cols]


print(f"Processing Slavic Analysis for {ENGINE.upper()}")
df_table = generate_stats_table()
if df_table is not None:
    df_table.to_csv(TABLE_OUTPUT, index=False)
    print(f"\nTable saved to {TABLE_OUTPUT}")