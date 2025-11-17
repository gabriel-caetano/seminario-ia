import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# Função para calcular Jensen-Shannon Distance para uma coluna
def calculate_js_distance(col1, col2, bins=50):
    # Remover valores NaN
    col1_clean = col1.dropna()
    col2_clean = col2.dropna()
    
    # Criar histogramas normalizados
    min_val = min(col1_clean.min(), col2_clean.min())
    max_val = max(col1_clean.max(), col2_clean.max())
    
    hist1, _ = np.histogram(col1_clean, bins=bins, range=(min_val, max_val))
    hist2, _ = np.histogram(col2_clean, bins=bins, range=(min_val, max_val))
    
    # Normalizar para criar distribuições de probabilidade
    p = hist1 / hist1.sum()
    q = hist2 / hist2.sum()
    
    # Adicionar pequeno valor para evitar divisão por zero
    p = p + 1e-10
    q = q + 1e-10
    
    # Normalizar novamente
    p = p / p.sum()
    q = q / q.sum()
    
    # Calcular Jensen-Shannon Distance
    return jensenshannon(p, q)

def plot_jsd(df_source_path, df_target_path):
    file_name = df_source_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(df_source_path.split('/')[:-1])
    file_extension = df_source_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")

    file_name = df_target_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(df_target_path.split('/')[:-1])
    file_extension = df_target_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")

    df_source = pd.read_csv(df_source_path)
    df_target = pd.read_csv(df_target_path)
    # Calcular JS Distance para cada atributo
    js_distances = {}
    columns = df_source.columns

    for col in columns:
        js_dist = calculate_js_distance(df_source[col], df_target[col])
        js_distances[col] = js_dist

    # Calcular JS Distance para o dataset completo (média ponderada)
    overall_js_distance = np.mean(list(js_distances.values()))
    js_distances['Dataset Completo'] = overall_js_distance

    # Criar DataFrame para visualização
    results_df = pd.DataFrame(list(js_distances.items()), columns=['Atributo', 'JS Distance'])

    # Definir cores pastéis baseadas nos valores
    def get_color(value):
        if value < 0.1:  # Muito similar
            return '#4CA64C'  # Verde pastel
        elif value < 0.3:  # Moderadamente similar
            return '#FFBF4C'  # Amarelo pastel
        else:  # Distinto
            return '#FE4D4D'  # Vermelho pastel

    colors = [get_color(val) for val in results_df['JS Distance']]

    # Criar gráfico de barras
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(results_df)), results_df['JS Distance'], color=colors)

    plt.xlabel('Atributos', fontsize=16, fontweight='bold')
    plt.ylabel('Distância Jensen-Shannon', fontsize=16, fontweight='bold')
    plt.title('Distância Jensen-Shannon entre Dataset Source e Target', 
            fontsize=18, fontweight='bold')
    plt.xticks(range(len(results_df)), results_df['Atributo'], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Ajustar limite superior para dar espaço extra acima da barra mais alta
    ymax = results_df['JS Distance'].max()
    # margem como uma fração razoável do valor máximo (ou um mínimo absoluto)
    headroom = max(0.03, ymax * 0.12)
    y_limit = min(1.0, ymax + headroom)
    plt.ylim(0, y_limit)

    # Adicionar valores nas barras (posicionando-os em função do y_limit para consistência)
    for i, (bar, value) in enumerate(zip(bars, results_df['JS Distance'])):
        y_text = bar.get_height() + (y_limit * 0.025)
        plt.text(bar.get_x() + bar.get_width()/2, y_text, f'{value:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Criar legenda
    legend_elements = [
        Patch(facecolor='#4CA64C', label='Muito Similar (JS < 0.1)'),
        Patch(facecolor='#FFBF4C', label='Moderadamente Similar (0.1 ≤ JS < 0.3)'),
        Patch(facecolor='#FE4D4D', label='Diferente (JS ≥ 0.3)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14, framealpha=0.9)

    plt.tight_layout()
    # Garantir que o diretório de saída exista
    out_dir = os.path.join(file_location, 'plot')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'{file_name}_jsd_plot.png'), dpi=300, bbox_inches='tight')
    # plt.show()

    # Imprimir resultados
    # print("=" * 60)
    # print("DISTÂNCIAS JENSEN-SHANNON")
    # print("=" * 60)
    # for idx, row in results_df.iterrows():
    #     status = "✓ Muito Similar" if row['JS Distance'] < 0.1 else ("⚠ Moderadamente Similar" if row['JS Distance'] < 0.3 else "✗ Diferente")
    #     print(f"{row['Atributo']:20s}: {row['JS Distance']:.6f} {status}")
    # print("=" * 60)