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
    plt.boxplot(data, tick_labels=['Origem', 'Destino'], widths=0.5)  # Adjust the width value as needed
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    
    plt.boxplot(data, tick_labels=['Origem', 'Destino'])
    plt.title(f'Box Plot of {column_name}')
    
    # plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_and_save_histograms(df_origem, df_destino, column_name, save_path, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    # Get the maximum value for both datasets to set consistent bins and range
    max_value = max(df_origem[column_name].max(), df_destino[column_name].max())
    min_value = min(df_origem[column_name].min(), df_destino[column_name].min())
    bins = np.linspace(min_value, max_value, 50)
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=df_origem[column_name], label='Origem', alpha=0.5)
    plt.title(f'Histogram of {column_name}\n(Origem)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_destino[column_name], label='Destino', alpha=0.5)
    plt.title(f'Histogram of {column_name}\n(Destino)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_and_save_barplot(df_origem, df_destino, column_name, save_path, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.margins(x=0.2)  # This adds space between the bars
    plt.rcParams.update({'font.size': 18, 'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
    
    value_counts_origem = df_origem[column_name].value_counts().sort_index()
    value_counts_destino = df_destino[column_name].value_counts().sort_index()
    
    x = np.arange(len(value_counts_origem))
    width = 0.35
    
    # Using slightly darker blue and yellow colors
    # Plot bars
    bars1 = plt.bar(x - width/2, value_counts_origem, width, label='Origem', color='#4682B4')  # Steel blue
    bars2 = plt.bar(x + width/2, value_counts_destino, width, label='Destino', color='#FFBF4C')  # Gold
    
    # Add value annotations above each bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    plt.title(f'Distribution of {column_name}')
    plt.xticks(x, value_counts_origem.index, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_distribution(csv_path_origem, csv_path_destino, figsize=(12, 6), ext='png'):
    """
    Create and save distribution plots comparing two datasets.
    
    Parameters:
    csv_path_origem (str): Path to the first CSV file (origem)
    csv_path_destino (str): Path to the second CSV file (destino)
    figsize (tuple): Figure size for plots (width, height)
    ext (str): File extension for saved plots (default: 'png')
    """
    # Load the CSV files
    df_origem = pd.read_csv(csv_path_origem)
    df_destino = pd.read_csv(csv_path_destino)

    # Create plots directory if it doesn't exist
    base_dir = os.path.dirname(csv_path_origem)
    plot_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(df_origem)
    
    # Plot distributions for numeric columns
    for col in numeric_cols:
        create_and_save_boxplot([df_origem[col], df_destino[col]], col, os.path.join(plot_dir, f'{col}_boxplot.{ext}'))
        
        create_and_save_histograms(df_origem, df_destino, col, os.path.join(plot_dir, f'{col}_histogram.{ext}'))
    
        # Plot distributions for categorical columns
    for col in categorical_cols:
        create_and_save_barplot(df_origem, df_destino, col, os.path.join(plot_dir, f'{col}_barplot.{ext}'))
    

if __name__ == "__main__":
    # Example usage
    print("Plotting distributions...")
    plot_distribution(
        'datasets/filled/boruta/age/dataset_filled_boruta_age>=60.csv'
        ,'datasets/filled/boruta/age/dataset_filled_boruta_age<60.csv'
    )
