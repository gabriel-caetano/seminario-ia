import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def fill_missing_values(dataset_path, target_column):
    file_name = dataset_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(dataset_path.split('/')[:-1])
    file_extension = dataset_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")
    # 1. Lê o arquivo
    dataset = pd.read_csv(dataset_path)
    # 2. Verifica colunas com valores ausentes e conta quantos faltam em cada uma
    missing_values_dict = dataset.isnull().sum()
    missing_values_dict = {col: int(count) for col, count in missing_values_dict.items() if count > 0}
    print("Missing values per column:", missing_values_dict)

    # 3. Identifica colunas com menos de 2% de valores ausentes
    total_rows = len(dataset)
    cols_less_2pct_missing = [col for col, count in missing_values_dict.items() if count / total_rows < 0.02]

    # 4. Remove registros com valores ausentes nessas colunas
    if cols_less_2pct_missing:
        dataset = dataset.dropna(subset=cols_less_2pct_missing)

    # 5. processed_dataset recebe o dataset atualizado
    processed_dataset = dataset
    missing_values_after = {col: int(count) for col, count in processed_dataset.isnull().sum().items() if count > 0}
    print("Missing values per column after processing:", missing_values_after)

    imputer = SimpleImputer(strategy='most_frequent') # Usando a moda para preencher valores ausentes
    processed_dataset['UPCR'] = imputer.fit_transform(processed_dataset[['UPCR']])
    imputer_bmi = SimpleImputer(strategy='median')
    processed_dataset['BMI'] = imputer_bmi.fit_transform(processed_dataset[['BMI']])
    
    processed_dataset.to_csv(f'{file_location}/filled/{file_name}_filled.csv', index=False)



if __name__ == "__main__":
    fill_missing_values('datasets/dataset.csv', 'CKD progression')