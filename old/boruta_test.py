import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.impute import SimpleImputer

def aplicar_boruta(caminho_csv, nome_coluna):

    # 1. Lê o arquivo
    dataset = pd.read_csv(caminho_csv)
    imputer = SimpleImputer(strategy='most_frequent')
    dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
    # 2. Definir X e y
    X = dataset.drop(columns=[nome_coluna]).values
    y = dataset[nome_coluna].values
    feature_names = dataset.drop(columns=[nome_coluna]).columns

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
    selected_columns = list(selected_features) + [nome_coluna]
    dataset_boruta = dataset[selected_columns]

    # 6. Salvar o novo arquivo
    dataset_boruta.to_csv('dataset_boruta.csv', index=False)

    print("\nNovo dataset salvo como 'dataset_boruta.csv'.")

aplicar_boruta('dataset.csv', 'CKD progression')