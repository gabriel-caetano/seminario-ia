import pandas as pd

def split_by_etiology(dataset_path='datasets/dataset.csv'):
    file_name = dataset_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(dataset_path.split('/')[:-1])
    file_extension = dataset_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")
    source_file = f'{file_name}_etiology12.csv'
    target_file = f'{file_name}_etiology34.csv'
    df = pd.read_csv(dataset_path)
    split_column = 'etiology of CKD'
    # etiology_counts = df[split_column].value_counts()
    # print("Rows per etiology:")
    # print(etiology_counts)
    # 2.0    451
    # 1.0    287
    # 3.0    216
    # 4.0    184


    source_df = df[(df[split_column] != 4) & (df[split_column] != 3)].copy()
    target_df = df[(df[split_column] == 4) | (df[split_column] == 3)].copy()

    # Remove the split column from both dataframes
    source_df = source_df.drop(columns=[split_column])
    target_df = target_df.drop(columns=[split_column])
    df = df.drop(columns=[split_column])

    # Save the new datasets to CSV files
    source_df.to_csv(f'{file_location}/etiology/{source_file}', index=False)
    target_df.to_csv(f'{file_location}/etiology/{target_file}', index=False)
    df.to_csv(f'{file_location}/etiology/{file_name}_etiologyfull.csv', index=False)

if __name__ == "__main__":
    split_by_etiology(dataset_path='datasets/dataset_filled_boruta.csv')