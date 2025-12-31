import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SUMMARY_CSV = 'final_evaluation_summary.csv'  # path to the summary CSV file
OUTPUT_DIR = 'results/plots'  # directory to save plots
os.makedirs(OUTPUT_DIR, exist_ok=True)


# plot safety profile - hallucinations (sub + ins) vs data loss (del)
def plot_safety_profile(df):
    df['Model'] = df['Model'].replace({'degraded': 'degraded (Baseline)'})

    col_unsafe = 'Unsafe (Hallucination)'
    col_safe = 'Safe (Data Loss)'

    df[col_unsafe] = df['Sub'] + df['Ins']
    df[col_safe] = df['Del']

    df_grouped = df.groupby('Model')[[col_unsafe, col_safe]].mean()
    df_grouped['Total_Errors'] = df_grouped[col_unsafe] + df_grouped[col_safe]

    sorted_models = df_grouped.sort_values('Total_Errors', ascending=False).index.tolist()

    if 'clean' in sorted_models:
        sorted_models.remove('clean')

    df_melt = df.melt(id_vars=['Model', 'Engine'],
                      value_vars=[col_unsafe, col_safe],
                      var_name='Error Type', value_name='Count')

    df_plot = df_melt[~df_melt['Model'].isin(['clean'])]

    plt.figure(figsize=(12, 7))

    palette = {col_unsafe: '#d62728', col_safe: '#1f77b4'}

    sns.barplot(data=df_plot, x='Model', y='Count', hue='Error Type',
                palette=palette, order=sorted_models)

    plt.title('Safety Profile: Hallucinations vs Data Loss', fontsize=16)
    plt.ylabel('Number of Characters', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    ax = plt.gca()
    for container in ax.containers:
        labels = [f'{val:.0f}' for val in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='edge', padding=0,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.1),
                     zorder=10)  # zorder ensures text sits ON TOP of the error bar

    plt.margins(y=0.01)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'safety_profile_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# load data and plot
if os.path.exists(SUMMARY_CSV):
    df = pd.read_csv(SUMMARY_CSV)
    plot_safety_profile(df)
else:
    print(f"Error: Could not find {SUMMARY_CSV}")
