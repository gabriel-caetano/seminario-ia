import pandas as pd
from scipy.stats import pearsonr

def calculate_pearson_correlation(dataset_path, column_y):
    df = pd.read_csv(dataset_path)
    correlations = {}
    for column in df.columns:
        if column != column_y:
            corr, _ = pearsonr(df[column], df[column_y])
            correlations[column] = corr
    return correlations

if __name__ == "__main__":
    dataset_path = 'datasets/filled/dataset_filled.csv'
    target_column = 'CKD progression'
    correlations = calculate_pearson_correlation(dataset_path, target_column)
    for feature, corr in correlations.items():
        print(f'Correlation between {feature} and {target_column}: {corr}')

