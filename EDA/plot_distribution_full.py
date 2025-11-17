import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def identify_column_types(df):
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        # Check if column contains only 0s and 1s (boolean)
        is_boolean = set(df[col].unique()) <= {0, 1}
        if is_boolean:
            categorical_cols.append(col)
            continue

        unique_values = sorted(df[col].unique())
        lequal_10 = len(unique_values) <= 10
        if lequal_10:
            categorical_cols.append(col)
            continue
        
        numeric_cols.append(col)
    
    return numeric_cols, categorical_cols

def create_and_save_boxplot(data, column_name, save_path, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.boxplot(data, tick_labels=[column_name], widths=0.5)  # Adjust the width value as needed
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    
    # plt.boxplot(data, tick_labels=['Origem', 'Destino'])
    plt.title(f'Box Plot of {column_name}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_and_save_histograms(df, column_name, save_path, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    # Get the maximum value for both datasets to set consistent bins and range
    # bins = np.linspace(min_value, max_value, 50)
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    
    sns.histplot(data=df[column_name], label='Origem', alpha=0.5)
    plt.title(f'Histogram of {column_name}\n(Origem)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_and_save_barplot(df, column_name, save_path, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

    value_counts = df[column_name].value_counts().sort_index()

    x = np.arange(len(value_counts))
    width = 0.5
    
    # Using slightly darker blue and yellow colors
    plt.bar(x, value_counts, width, label='Origem', color='#4682B4')  # Steel blue
    plt.title(f'Distribution of {column_name}')
    plt.xticks(x, value_counts.index, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_and_save_barplot_pct(df, column_name, save_path, figsize=(6, 6)):
    """Create and save a barplot showing percentages for the given categorical column."""
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

    value_counts = df[column_name].value_counts().sort_index()
    total = value_counts.sum()
    pct = (value_counts / total) * 100

    x = np.arange(len(pct))
    width = 0.6

    plt.bar(x, pct, width, color='#4682B4')
    plt.title(f'Distribution of {column_name} (percent)')
    plt.xticks(x, pct.index, rotation=45)
    plt.ylabel('Percent (%)')

    # Set y-limit with a little headroom
    ymax = max(pct.max(), 10)
    plt.ylim(0, min(100, ymax * 1.15))

    # Annotate bars with percentage values
    for xi, val in zip(x, pct.values):
        plt.text(xi, val + (plt.ylim()[1] * 0.02), f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_distribution_full(csv_path, figsize=(6, 6), ext='png'):
    """
    Create and save distribution plots comparing two datasets.
    
    Parameters:
    csv_path (str): Path to the CSV file
    figsize (tuple): Figure size for plots (width, height)
    ext (str): File extension for saved plots (default: 'png')
    """
    # Load the CSV files
    df = pd.read_csv(csv_path)

    # Create plots directory if it doesn't exist
    base_dir = os.path.dirname(csv_path)
    plot_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(df)

    # Plot distributions for numeric columns
    for col in numeric_cols:
        create_and_save_boxplot([df[col]], col, os.path.join(plot_dir, f'{col}_boxplot.{ext}'))

        create_and_save_histograms(df, col, os.path.join(plot_dir, f'{col}_histogram.{ext}'))

        # Plot distributions for categorical columns
    for col in categorical_cols:
        create_and_save_barplot(df, col, os.path.join(plot_dir, f'{col}_barplot.{ext}'))
        create_and_save_barplot_pct(df, col, os.path.join(plot_dir, f'{col}_barplot_pct.{ext}'))
    

if __name__ == "__main__":
    # Example usage
    print("Plotting distributions...")
    plot_distribution_full(
        'datasets/filled/boruta/dataset_filled_boruta.csv'
    )
