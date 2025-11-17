import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.impute import SimpleImputer

def apply_boruta(dataset_path, target_column):
    file_name = dataset_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(dataset_path.split('/')[:-1])
    file_extension = dataset_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")
    # 1. Lê o arquivo
    dataset = pd.read_csv(dataset_path)

    # 2. Definir X e y
    X = dataset.drop(columns=[target_column]).values
    y = dataset[target_column].values
    feature_names = dataset.drop(columns=[target_column]).columns

    # 3. Aplicar Boruta
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    boruta = BorutaPy(rf, random_state=42)
    boruta.fit(X, y)

    # 4. Identificar as features selecionadas
    selected_features = feature_names[boruta.support_]
    removed_features = feature_names[~boruta.support_]

    print("Campos removidos:")
    for feature in removed_features:
        print(f"- {feature}")

    # 5. Criar novo dataset com apenas as features selecionadas + target
    selected_columns = list(selected_features) + [target_column]
    dataset_boruta = dataset[selected_columns]

    # 6. Salvar o novo arquivo
    dataset_boruta.to_csv(f'{file_location}/boruta/{file_name}_boruta.csv', index=False)

    print(f"\nNovo dataset salvo como '{file_location}/boruta/{file_name}_boruta.csv'.")


if __name__ == "__main__":
    apply_boruta('datasets/dataset.csv', 'CKD progression')