import pandas as pd

def split_by_etiology(dataset_path='datasets/dataset.csv'):
    file_name = dataset_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(dataset_path.split('/')[:-1])
    file_extension = dataset_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")
    source_file = f'{file_name}_etiology123.csv'
    target_file = f'{file_name}_etiology4.csv'
    df = pd.read_csv(dataset_path)
    split_column = 'etiology of CKD'
    # etiology_counts = df[split_column].value_counts()
    # print("Rows per etiology:")
    # print(etiology_counts)
    # 2.0    451
    # 1.0    287
    # 3.0    216
    # 4.0    184


    source_df = df[df[split_column] != 4].copy()

    target_df = df[df[split_column] == 4].copy()

    # Save the new datasets to CSV files
    source_df.to_csv(f'{file_location}/{source_file}', index=False)
    target_df.to_csv(f'{file_location}/{target_file}', index=False)

if __name__ == "__main__":
    split_by_etiology(dataset_path='datasets/dataset_filled_boruta.csv')