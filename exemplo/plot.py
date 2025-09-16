import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os datasets
try:
    df_adultos = pd.read_csv('../dataset_progression/adultos.csv')
    df_idosos = pd.read_csv('../dataset_progression/idosos.csv')
    df_dataset = pd.read_csv('../dataset_progression/dataset.csv')

    # Adicionar uma coluna de identificação em cada dataframe
    df_adultos['grupo'] = 'Adultos (<60 anos)'
    df_idosos['grupo'] = 'Idosos (>=60 anos)'
    df_dataset['grupo'] = 'Completo'

    # Combinar os dataframes para facilitar a plotagem
    df_combined = pd.concat([df_adultos, df_idosos, df_dataset], ignore_index=True)

    # Definir a paleta de cores
    palette = {'Adultos (<60 anos)': 'skyblue', 'Idosos (>=60 anos)': 'salmon', 'Completo': 'lightgrey'}

    # --- Visualizações de Variáveis Numéricas (Boxplots) ---
    numerical_cols = ['age', 'SBP', 'Hb', 'Alb', 'eGFR', 'UPCR']
    # Save 3 boxplots per row in each image inside 'plots/' folder, with larger text
    for i in range(0, len(numerical_cols), 3):
        fig, axes = plt.subplots(1, 3, figsize=(36, 12))
        for j in range(3):
            if i + j < len(numerical_cols):
                col = numerical_cols[i + j]
                sns.boxplot(x='grupo', y=col, data=df_combined, ax=axes[j], palette=palette, order=['Adultos (<60 anos)', 'Idosos (>=60 anos)', 'Completo'], hue='grupo', legend=False)
                axes[j].set_title(f'Distribuição de {col}', fontsize=36)
                axes[j].set_xlabel('Grupo', fontsize=32)
                axes[j].set_ylabel(col, fontsize=32)
                axes[j].tick_params(axis='x', labelsize=28, rotation=10)
                axes[j].tick_params(axis='y', labelsize=28)
            else:
                axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(f'plots/boxplots_{i//3+1}.png')
        plt.close(fig)


    # --- Visualizações de Variáveis Categóricas (Gráficos de Barras) ---
    categorical_cols = ['etiology of CKD', 'proteinuria', 'CKD progression']
    # Save 3 barplots per row in each image inside 'plots/' folder, with larger text
    for i in range(0, len(categorical_cols), 3):
        fig, axes = plt.subplots(1, 3, figsize=(36, 12))
        for j in range(3):
            if i + j < len(categorical_cols):
                col = categorical_cols[i + j]
                sns.countplot(x=col, hue='grupo', data=df_combined, ax=axes[j], palette=palette, hue_order=['Adultos (<60 anos)', 'Idosos (>=60 anos)', 'Completo'])
                axes[j].set_title(f'Contagem por {col}', fontsize=36)
                axes[j].set_xlabel(col, fontsize=32)
                axes[j].set_ylabel('Contagem', fontsize=32)
                axes[j].legend(title='Grupo', fontsize=28, title_fontsize=32)
                axes[j].tick_params(axis='x', labelsize=28)
                axes[j].tick_params(axis='y', labelsize=28)
            else:
                axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(f'plots/barplots_{i//3+1}.png')
        plt.close(fig)

except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Verifique se os arquivos 'adultos.csv', 'idosos.csv' e 'dataset.csv' estão no diretório correto.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")