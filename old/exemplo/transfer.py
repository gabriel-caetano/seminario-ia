
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
tf.config.set_visible_devices([], 'GPU')
# Definindo a semente para reprodutibilidade
tf.random.set_seed(42)
keras = tf.keras
scaler = StandardScaler()

# --- 1. Preparação dos Dados ---
source_df = pd.read_csv('dataset_source.csv')

X_source = source_df.drop('CKD progression', axis=1)
y_source = source_df['CKD progression']
X_train_source, X_val_source, y_train_source, y_val_source = train_test_split(
    X_source, y_source, test_size=0.3, random_state=42
)
X_train_scaled_source = scaler.fit_transform(X_train_source)
X_val_scaled_source = scaler.transform(X_val_source)

base_model_source = keras.models.Sequential([
    # Camada de entrada com a quantidade de features do nosso dataset
    keras.layers.InputLayer(input_shape=(X_train_scaled_source.shape[1],)),
    
    # Camadas ocultas com ativação ReLU
    keras.layers.Dense(16, activation='relu', name='camada_oculta_1'),
    keras.layers.Dense(8, activation='relu', name='camada_oculta_2'),
    
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
    epochs=20,
    validation_data=(X_val_scaled_source, y_val_source),
    verbose=0 
)
print("Treinamento concluído!")

# --- 5. Avaliação do Modelo Base ---
loss, accuracy = base_model_source.evaluate(X_val_scaled_source, y_val_source, verbose=0)
print(f"\nAcurácia do modelo base no conjunto source: {accuracy:.2%}")
print(f"\nPerda do modelo base no conjunto source: {loss:.2%}")

# --- Métricas detalhadas do modelo base (fonte) ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred_source = (base_model_source.predict(X_val_scaled_source) > 0.5).astype(int)
print("\nMétricas detalhadas no conjunto de validação (Fonte):")
print(f"Acurácia : {accuracy_score(y_val_source, y_pred_source):.2%}")
print(f"Precisão : {precision_score(y_val_source, y_pred_source, zero_division=0):.2%}")
print(f"Recall   : {recall_score(y_val_source, y_pred_source, zero_division=0):.2%}")
print(f"F1-score : {f1_score(y_val_source, y_pred_source, zero_division=0):.2%}")

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
X_target = target_df.drop('CKD progression', axis=1)
y_target = target_df['CKD progression']

X_train_target, X_val_target, y_train_target, y_val_target = train_test_split(
    X_target, y_target, test_size=0.2, random_state=42
)

# IMPORTANTE: Usamos o MESMO scaler treinado nos dados fonte.
# Não usamos fit_transform, apenas transform. Isso garante que a escala dos dados seja consistente.
X_train_scaled_target = scaler.transform(X_train_target)
X_val_scaled_target = scaler.transform(X_val_target)

# --- 7. Avaliação do modelo com os dados alvo (Antes do Retreino) ---

# --- Avaliação antes do transfer learning ---
loss_before, accuracy_before = base_model_source.evaluate(X_val_scaled_target, y_val_target, verbose=0)
y_pred_before = (base_model_source.predict(X_val_scaled_target) > 0.5).astype(int)
prec_before = precision_score(y_val_target, y_pred_before, zero_division=0)
rec_before = recall_score(y_val_target, y_pred_before, zero_division=0)
f1_before = f1_score(y_val_target, y_pred_before, zero_division=0)
print("\n--- Avaliação no domínio ALVO *antes* do retreino ---")
print(f"Acurácia : {accuracy_before:.2%}")
print(f"Perda    : {loss_before:.2%}")
print(f"Precisão : {prec_before:.2%}")
print(f"Recall   : {rec_before:.2%}")
print(f"F1-score : {f1_before:.2%}")

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

# --- Avaliação depois do transfer learning ---
loss_after, accuracy_after = base_model_source.evaluate(X_val_scaled_target, y_val_target, verbose=0)
y_pred_after = (base_model_source.predict(X_val_scaled_target) > 0.5).astype(int)
prec_after = precision_score(y_val_target, y_pred_after, zero_division=0)
rec_after = recall_score(y_val_target, y_pred_after, zero_division=0)
f1_after = f1_score(y_val_target, y_pred_after, zero_division=0)
print("\n--- Avaliação no domínio ALVO *depois* do retreino ---")
print(f"Acurácia : {accuracy_after:.2%}")
print(f"Perda    : {loss_after:.2%}")
print(f"Precisão : {prec_after:.2%}")
print(f"Recall   : {rec_after:.2%}")
print(f"F1-score : {f1_after:.2%}")

# --- 10. Comparação e Visualização ---

# --- Comparação detalhada das métricas (Target) ---
print("\n--- Melhora das métricas após Transfer Learning (Target) ---")
print(f"Acurácia : {accuracy_after - accuracy_before:+.2%}")
print(f"Perda    : {loss_before - loss_after:+.2%}")
print(f"Precisão : {prec_after - prec_before:+.2%}")
print(f"Recall   : {rec_after - rec_before:+.2%}")
print(f"F1-score : {f1_after - f1_before:+.2%}")

# --- Comparação entre treino e validação após transfer learning ---
train_acc_t = history_target.history['accuracy'][-1]
val_acc_t = history_target.history['val_accuracy'][-1]
train_loss_t = history_target.history['loss'][-1]
val_loss_t = history_target.history['val_loss'][-1]

print("\n--- Comparação Final Treino vs Validação (Transfer Learning) ---")
print(f"Acurácia final - Treino: {train_acc_t:.2%} | Validação: {val_acc_t:.2%}")
print(f"Perda final    - Treino: {train_loss_t:.4f} | Validação: {val_loss_t:.4f}")
if train_acc_t - val_acc_t > 0.05:
    print("\nPossível overfitting detectado após transfer learning: a acurácia de treino está significativamente maior que a de validação.")
elif val_loss_t > train_loss_t * 1.2:
    print("\nPossível overfitting detectado após transfer learning: a perda de validação está significativamente maior que a de treino.")
else:
    print("\nNão há sinais claros de overfitting após transfer learning.")
print(f"Melhora de acurácia com Transfer Learning: {accuracy_after - accuracy_before:+.2%}")
print(f"Melhora de perda com Transfer Learning: {loss_before - loss_after:+.2%}")

# pd.DataFrame(history_target.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.title("Curvas de Aprendizado do RETREINO (Dados Target)")
# plt.xlabel("Epochs")
# plt.show()



# Métrica         Antes     Depois    Melhora
# Acurácia       82.50%     83.75%     +1.25%
# Perda          0.9472     0.3938    +0.5534
# Precisão       75.00%     58.82%    -16.18%
# Recall         18.75%     62.50%    +43.75%
# F1-score       30.00%     60.61%    +30.61%

# Métrica         Antes     Depois    Melhora   base line
# Acurácia       80.00%     83.75%     +3.75%   86.25%
# Perda          0.7874     0.4214    +0.3660   0.4793
# Precisão       50.00%     58.82%     +8.82%   61.90%
# Recall          6.25%     62.50%    +56.25%   81.25%
# F1-score       11.11%     60.61%    +49.49%   70.27%