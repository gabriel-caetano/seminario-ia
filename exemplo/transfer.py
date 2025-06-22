import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Definindo a semente para reprodutibilidade
tf.random.set_seed(42)
keras = tf.keras
scaler = StandardScaler()

# --- 1. Preparação dos Dados ---
source_df = pd.read_csv('dataset_source.csv')

X_source = source_df.drop('death', axis=1)
y_source = source_df['death']
X_train_source, X_val_source, y_train_source, y_val_source = train_test_split(
    X_source, y_source, test_size=0.2, random_state=42
)
X_train_scaled_source = scaler.fit_transform(X_train_source)
X_val_scaled_source = scaler.transform(X_val_source)

base_model_source = keras.models.Sequential([
    # Camada de entrada com a quantidade de features do nosso dataset
    keras.layers.InputLayer(input_shape=(X_train_scaled_source.shape[1],)),
    
    # Camadas ocultas com ativação ReLU
    keras.layers.Dense(32, activation='relu', name='camada_oculta_1'),
    keras.layers.Dense(16, activation='relu', name='camada_oculta_2'),
    keras.layers.Dense(8, activation='relu', name='camada_oculta_3'),
    
    # Camada de saída: 1 neurônio porque é classificação binária (morreu/não morreu)
    # Ativação Sigmoid para retornar uma probabilidade entre 0 e 1.
    keras.layers.Dense(1, activation='sigmoid', name='camada_saida')
])

# --- 3. Compilação do Modelo ---
base_model_source.compile(
    loss='binary_crossentropy',      # função para classificação binária
    optimizer=keras.optimizers.Adam(learning_rate=0.01), # valor padrão .001
    metrics=['accuracy']
)

print("\n--- Arquitetura do Modelo Base ---")
base_model_source.summary()
print("-" * 25)

# --- 4. Treinamento do Modelo Base ---
print("\n--- Treinando o Modelo Base com os dados fonte (pacientes idosos) ---")
history = base_model_source.fit(
    X_train_scaled_source,
    y_train_source,
    epochs=100,
    validation_data=(X_val_scaled_source, y_val_source),
    verbose=0 
)
print("Treinamento concluído!")

# --- 5. Avaliação do Modelo Base ---
loss, accuracy = base_model_source.evaluate(X_val_scaled_source, y_val_source, verbose=0)
print(f"\nAcurácia do modelo base no conjunto source: {accuracy:.2%}")
print(f"\nPerda do modelo base no conjunto source: {loss:.2%}")

# --- 6. Visualização do Treinamento ---
# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1) # set the y-axis range to [0, 1]
# plt.title("Curvas de Aprendizado do Modelo Base source")
# plt.xlabel("Epochs")
# plt.show()

print("\n--- Iniciando Transfer Learning para o Domínio Alvo (pacientes jovens) ---")

# --- 6. Preparação dos Dados Alvo ---
target_df = pd.read_csv('dataset_target.csv')
X_target = target_df.drop('death', axis=1)
y_target = target_df['death']

X_train_target, X_val_target, y_train_target, y_val_target = train_test_split(
    X_target, y_target, test_size=0.2, random_state=42
)

# IMPORTANTE: Usamos o MESMO scaler treinado nos dados fonte.
# Não usamos fit_transform, apenas transform. Isso garante que a escala dos dados seja consistente.
X_train_scaled_target = scaler.transform(X_train_target)
X_val_scaled_target = scaler.transform(X_val_target)

# --- 7. Avaliação do modelo com os dados alvo (Antes do Retreino) ---
loss_before, accuracy_before = base_model_source.evaluate(X_val_scaled_target, y_val_target, verbose=0)
print(f"Acurácia no domínio ALVO *antes* do retreino: {accuracy_before:.2%}")
print(f"\nPerda no domínio ALVO *antes* do retreino: {loss_before:.2%}")

# --- 8. Retreinamento do Modelo (Transfer Learning) ---
print("\n--- Retreinando o modelo com os dados do domínio alvo... ---")
base_model_source.layers[0].trainable = False
base_model_source.layers[1].trainable = False

base_model_source.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.01), # Redefinindo taxa de aprendizado
    metrics=['accuracy']
)

history_target = base_model_source.fit(
    X_train_scaled_target,
    y_train_target,
    epochs=10,
    validation_data=(X_val_scaled_target, y_val_target),
    verbose=0
)
print("Retreinamento concluído!")

# --- 9. Avaliação Final (Depois do Retreino) ---
loss_after, accuracy_after = base_model_source.evaluate(X_val_scaled_target, y_val_target, verbose=0)
print(f"\nAcurácia no domínio ALVO *depois* do retreino: {accuracy_after:.2%}")
print(f"\nPerda no domínio ALVO *depois* do retreino: {loss_after:.2%}")

# --- 10. Comparação e Visualização ---
print("\n--- Resumo do Resultado ---")
print(f"Acurácia inicial no alvo: {accuracy_before:.2%}")
print(f"Perda inicial no alvo: {loss_before:.2%}")
print(f"Acurácia final no alvo:   {accuracy_after:.2%}")
print(f"Perda final no alvo:   {loss_after:.2%}")
print(f"Melhora de acurácia com Transfer Learning: {accuracy_after - accuracy_before:+.2%}")
print(f"Melhora de perda com Transfer Learning: {loss_before - loss_after:+.2%}")

# pd.DataFrame(history_target.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.title("Curvas de Aprendizado do RETREINO (Dados Target)")
# plt.xlabel("Epochs")
# plt.show()