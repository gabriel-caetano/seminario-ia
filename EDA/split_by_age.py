import pandas as pd

def split_by_age(age=60, dataset_path='datasets/dataset.csv'):
    file_name = dataset_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(dataset_path.split('/')[:-1])
    file_extension = dataset_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")
    source_file = f'{file_name}_age>={age}.csv'
    target_file = f'{file_name}_age<{age}.csv'
    df = pd.read_csv(dataset_path)
    split_column = 'age'
    source_df = df[df[split_column] >= age].copy()

    target_df = df[df[split_column] < age].copy()

    # Save the new datasets to CSV files
    source_df.to_csv(f'{file_location}/{source_file}', index=False)
    target_df.to_csv(f'{file_location}/{target_file}', index=False)

if __name__ == "__main__":
    split_by_age(dataset_path='datasets/dataset_filled_boruta.csv')