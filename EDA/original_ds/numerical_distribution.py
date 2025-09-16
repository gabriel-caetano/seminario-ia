import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carregar o arquivo Excel
file_path = 'ROUTE_proteinuria_dataset.csv'
df = pd.read_csv(file_path)

# Configurações de estilo para os gráficos
sns.set_theme(style="whitegrid")

# Lista de variáveis para plotar
variables_to_plot = ['age', 'BMI', 'eGFR', 'UPCR', 'SBP', 'Cr']
titles = {
    'age': 'Idade (anos)',
    'BMI': 'Índice de Massa Corporal (IMC)',
    'eGFR': 'Taxa de Filtração Glomerular Estimada (TFG)',
    'UPCR': 'Relação Proteína/Creatinina Urinária',
    'SBP': 'Pressão Arterial Sistólica (PAS)',
    'Cr': 'Creatinina Sérica (mg/dL)'
}

# Criar uma figura para todos os gráficos
fig, axes = plt.subplots(len(variables_to_plot), 2, figsize=(14, 20), gridspec_kw={'width_ratios': [3, 1]})
fig.suptitle('Análise de Distribuição de Variáveis Numéricas', fontsize=20, y=1.02)

for i, var in enumerate(variables_to_plot):
    # Remover NaNs para a variável atual
    data = df[var].dropna()
    
    # Histograma
    sns.histplot(data, kde=True, ax=axes[i, 0], bins=30)
    axes[i, 0].set_title(f'Histograma de {titles[var]}', fontsize=14)
    axes[i, 0].set_xlabel('')
    axes[i, 0].set_ylabel('Frequência')
    
    # Boxplot
    sns.boxplot(y=data, ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot de {titles[var]}', fontsize=14)
    axes[i, 1].set_ylabel('')
    axes[i, 1].set_xlabel('')

plt.tight_layout(rect=[0, 0, 1, 1])
# Criar o diretório se não existir
os.makedirs('./plots', exist_ok=True)

# Salvar a figura
fig.savefig('./plots/distribuicao_variaveis_numericas.png', bbox_inches='tight')