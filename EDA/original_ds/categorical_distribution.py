import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carregar o arquivo Excel
file_path = 'ROUTE_proteinuria_dataset.csv'
df = pd.read_csv(file_path)

# Configurações de estilo para os gráficos
sns.set_theme(style="whitegrid")

# Lista de variáveis categóricas para plotar
categorical_vars = [
    'gender', 'etiology of CKD', 'CKD_stage', 
    'hypertension', 'diabetes', 'proteinuria'
]

# Mapeamento de nomes para os títulos e rótulos dos gráficos
titles = {
    'gender': 'Distribuição por Gênero',
    'etiology of CKD': 'Distribuição por Etiologia da DRC',
    'CKD_stage': 'Distribuição por Estágio da DRC',
    'hypertension': 'Prevalência de Hipertensão',
    'diabetes': 'Prevalência de Diabetes',
    'proteinuria': 'Prevalência de Proteinúria'
}

# Mapeamento dos valores para rótulos mais claros
df_plot = df.copy()
df_plot['gender'] = df_plot['gender'].map({1: 'Masculino', 2: 'Feminino'})
df_plot['hypertension'] = df_plot['hypertension'].map({0: 'Não', 1: 'Sim'})
df_plot['diabetes'] = df_plot['diabetes'].map({0: 'Não', 1: 'Sim'})
df_plot['proteinuria'] = df_plot['proteinuria'].map({0: 'Não', 1: 'Sim'})
df_plot['CKD_stage'] = 'Estágio ' + df['CKD_stage'].astype(int).astype(str)
df_plot['etiology of CKD'] = 'Etiologia ' + df['etiology of CKD'].astype(int).astype(str)


# Criar uma figura separada para cada atributo
for var in categorical_vars:
    plt.figure(figsize=(7, 5))
    order = df_plot[var].value_counts().index
    ax = sns.countplot(x=var, hue=var, data=df_plot, palette='viridis', order=order, legend=False)
    # Ajustar o limite superior do eixo y para dar mais espaço aos rótulos
    y_max = max([p.get_height() for p in ax.patches]) * 1.15
    ax.set_ylim(0, y_max)
    ax.set_title(titles[var], fontsize=16)
    ax.set_xlabel('')
    ax.set_ylabel('Contagem de Pacientes', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    total = len(df_plot[var].dropna())
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(f'{p.get_height()}\n({percentage})', (x, y), ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    os.makedirs('./plots/categorical/', exist_ok=True)
    plt.savefig(f'./plots/categorical/distribuicao_{var}.png', bbox_inches='tight')
    plt.close()
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()
fig.suptitle('Análise de Distribuição de Variáveis Categóricas', fontsize=22, y=1.03)

for i, var in enumerate(categorical_vars):
    # Ordenar categorias pela contagem para melhor visualização
    order = df_plot[var].value_counts().index
    
    # Criar o gráfico de barras
    sns.countplot(x=var, hue=var, data=df_plot, ax=axes[i], palette='viridis', order=order, legend=False)
    
    axes[i].set_title(titles[var], fontsize=16)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Contagem de Pacientes', fontsize=12)
    axes[i].tick_params(axis='x', rotation=0) # Garante que os rótulos não fiquem rotacionados
    
    # Adicionar rótulos de contagem e percentual
    total = len(df_plot[var].dropna())
    for p in axes[i].patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        axes[i].annotate(f'{p.get_height()}\n({percentage})', (x, y), ha='center', va='bottom', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 1])
# Criar o diretório se não existir
os.makedirs('./plots', exist_ok=True)

# Salvar a figura
fig.savefig('./plots/distribuicao_variaveis_categoricas.png', bbox_inches='tight')