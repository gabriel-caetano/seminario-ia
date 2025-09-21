import pandas as pd

def split_by_stage(dataset_path='datasets/dataset.csv'):
    file_name = dataset_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(dataset_path.split('/')[:-1])
    file_extension = dataset_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")
    source_file = f'{file_name}_stage345.csv'
    target_file = f'{file_name}_stage2.csv'
    df = pd.read_csv(dataset_path)
    split_column = 'CKD_stage'
    stage_counts = df[split_column].value_counts()
    print("Rows per stage:")
    print(stage_counts)
    # 3.0    455
    # 4.0    354
    # 5.0    198
    # 2.0     92


    source_df = df[df[split_column] != 2].copy()

    target_df = df[df[split_column] == 2].copy()

    # Save the new datasets to CSV files
    source_df.to_csv(f'{file_location}/{source_file}', index=False)
    target_df.to_csv(f'{file_location}/{target_file}', index=False)

if __name__ == "__main__":
    split_by_stage(dataset_path='datasets/dataset_filled_boruta.csv')