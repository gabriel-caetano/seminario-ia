import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definindo a semente para reprodutibilidade
tf.random.set_seed(42)
keras = tf.keras
scaler = StandardScaler()

# --- 1. Preparação dos Dados ---
target_df = pd.read_csv('dataset_target.csv')

# O modelo será treinado do zero nos dados "alvo".
X_target = target_df.drop('death', axis=1)
y_target = target_df['death']
X_train_target, X_val_target, y_train_target, y_val_target = train_test_split(
    X_target, y_target, test_size=0.2, random_state=42
)
X_train_scaled_target = scaler.fit_transform(X_train_target)
X_val_scaled_target = scaler.transform(X_val_target)

base_model_target = keras.models.Sequential([
    # Camada de entrada com a quantidade de features do nosso dataset
    keras.layers.InputLayer(shape=(X_train_scaled_target.shape[1],)),
    
    # Camadas ocultas com ativação ReLU
    keras.layers.Dense(32, activation='relu', name='camada_oculta_1'),
    keras.layers.Dense(16, activation='relu', name='camada_oculta_2'),
    keras.layers.Dense(8, activation='relu', name='camada_oculta_3'),
    
    # Camada de saída: 1 neurônio porque é classificação binária (morreu/não morreu)
    # Ativação Sigmoid para retornar uma probabilidade entre 0 e 1.
    keras.layers.Dense(1, activation='sigmoid', name='camada_saida')
])

# --- 3. Compilação do Modelo ---

# Compilar o modelo define a função de perda, o otimizador e as métricas.
base_model_target.compile(
    loss='binary_crossentropy',      # Ideal para classificação binária
    optimizer=keras.optimizers.Adam(learning_rate=0.01), # Otimizador popular e eficiente
    metrics=['accuracy']             # Métrica para acompanhar o desempenho
)

print("\n--- Arquitetura do Modelo Base ---")
base_model_target.summary()
print("-" * 25)

# --- 4. Treinamento do Modelo Base ---
print("\n--- Treinando o Modelo Base (nos dados alvo) ---")
history = base_model_target.fit(
    X_train_scaled_target,
    y_train_target,
    epochs=50, # Número de vezes que o modelo verá todo o dataset
    validation_data=(X_val_scaled_target, y_val_target),
    verbose=0 # Usamos verbose=0 para não poluir a saída, mas você pode usar 1 para ver o progresso
)
print("Treinamento concluído!")

# --- 5. Avaliação do Modelo Base ---

loss, accuracy = base_model_target.evaluate(X_val_scaled_target, y_val_target, verbose=0)
print(f"\nAcurácia do modelo base no conjunto target: {accuracy:.2%}")
print(f"\nPerda do modelo base no conjunto target: {loss:.2%}")

# --- 6. Visualização do Treinamento ---

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the y-axis range to [0, 1]
plt.title("Curvas de Aprendizado do Modelo Base target")
plt.xlabel("Epochs")
plt.show()