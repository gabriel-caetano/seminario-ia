import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

    # # 3. Identifica colunas com menos de 2% de valores ausentes
    # total_rows = len(dataset)
    # print("Total rows in dataset:", total_rows)
    # cols_less_2pct_missing = [col for col, count in missing_values_dict.items() if count / total_rows < 0.02]
    # print("Columns with less than 2% missing values:", cols_less_2pct_missing)
    # # 4. Remove registros com valores ausentes nessas colunas
    # if cols_less_2pct_missing:
    #     dataset = dataset.dropna(subset=cols_less_2pct_missing)

    # 5. processed_dataset recebe o dataset atualizado
    processed_dataset = dataset
    missing_values_after = {col: int(count) for col, count in processed_dataset.isnull().sum().items() if count > 0}
    print("Missing values per column after processing:", missing_values_after)

    for column in missing_values_after.keys():
        flag_column = f'{column}_missing_flag'
        processed_dataset[flag_column] = processed_dataset[column].isnull().astype(int)

    for column in missing_values_after.keys():
        df_missing = processed_dataset[processed_dataset[column].isnull()]
        df_not_missing = processed_dataset[processed_dataset[column].notnull()]

        if df_missing.empty:
            continue

        X_train = df_not_missing.drop(columns=[column, target_column])
        y_train = df_not_missing[column]
        X_pred = df_missing.drop(columns=[column, target_column])

        cols_with_missing_in_X_train = X_train.columns[X_train.isnull().any()].tolist()
        cols_with_missing_in_X_pred = X_pred.columns[X_pred.isnull().any()].tolist()
        cols_to_drop = list(set(cols_with_missing_in_X_train) | set(cols_with_missing_in_X_pred))
        X_train = X_train.drop(columns=cols_to_drop)
        X_pred = X_pred.drop(columns=cols_to_drop)

        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)

        predicted_values = rf_regressor.predict(X_pred)

        processed_dataset.loc[processed_dataset[column].isnull(), column] = predicted_values
    
    processed_dataset.to_csv(f'{file_location}/filled/{file_name}_rfregressor_and_missing_flags_filled.csv', index=False)



if __name__ == "__main__":
    fill_missing_values('datasets/dataset.csv', 'CKD progression')