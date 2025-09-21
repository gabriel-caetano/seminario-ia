import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurações de visualização
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 12

# Carregar o dataset
df = pd.read_csv('ROUTE_proteinuria_dataset.csv')

# --- Passo 1: Limpeza e Preparação ---
# Remover colunas que são claramente inúteis
cols_to_drop = ['ID'] + [col for col in df.columns if 'Unnamed' in col]
df_cleaned = df.drop(columns=cols_to_drop)

print("--- Verificação de Dados Faltantes (NaNs) por Coluna ---")
missing_values = df_cleaned.isnull().sum()
print(missing_values[missing_values > 0])
print("\nAnálise iniciada. Gerando gráficos...")

# --- Gráficos da Análise ---
plt.figure(figsize=(20, 15))

# --- Gráfico 1: Análise da Variável-Alvo (CKD progression) ---
plt.subplot(2, 2, 1)
progression_counts = df_cleaned['CKD progression'].value_counts()
plt.pie(progression_counts, labels=['Não Progrediu', 'Progrediu'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
plt.title('Distribuição da Progressão da DRC (Variável-Alvo)', fontweight='bold')
plt.ylabel('')

# --- Gráfico 2: Relação entre Estágio da DRC e Progressão ---
plt.subplot(2, 2, 2)
# Criar uma tabela de contingência e normalizar para obter percentuais
crosstab_stage = pd.crosstab(df_cleaned['CKD_stage'], df_cleaned['CKD progression'])
crosstab_stage_norm = crosstab_stage.div(crosstab_stage.sum(1).astype(float), axis=0)
crosstab_stage_norm.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#66b3ff','#ff9999'])
plt.title('Progressão da DRC por Estágio da Doença', fontweight='bold')
plt.xlabel('Estágio da DRC')
plt.ylabel('Proporção de Pacientes')
plt.xticks(rotation=0)
plt.legend(['Não Progrediu', 'Progrediu'])

# --- Gráfico 3: Distribuição de Idade vs. Progressão ---
plt.subplot(2, 2, 3)
sns.histplot(data=df_cleaned, x='age', hue='CKD progression', multiple='stack', palette=['#66b3ff','#ff9999'], kde=True)
plt.title('Distribuição de Idade por Status de Progressão', fontweight='bold')
plt.xlabel('Idade (anos)')
plt.ylabel('Contagem de Pacientes')

# --- Gráfico 4: Distribuição de eGFR vs. Progressão ---
plt.subplot(2, 2, 4)
sns.boxplot(data=df_cleaned, x='CKD progression', y='eGFR', palette=['#66b3ff','#ff9999'])
plt.title('eGFR por Status de Progressão', fontweight='bold')
plt.xlabel('Progressão da DRC (0 = Não, 1 = Sim)')
plt.ylabel('eGFR (Taxa de Filtração Glomerular)')
plt.xticks([0, 1], ['Não Progrediu', 'Progrediu'])


plt.tight_layout()
plt.show()

# --- Gráfico 5: Heatmap de Correlação ---
plt.figure(figsize=(16, 12))
numeric_cols = df_cleaned.select_dtypes(include=np.number)
corr_matrix = numeric_cols.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Heatmap de Correlação entre Atributos Numéricos', fontweight='bold')
plt.show()