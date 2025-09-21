from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HIDDEN_LAYERS=[4, 2]
ACTIVATION='relu'
LEARNING_RATE=0.01
LOSS='binary_crossentropy'

class MLP:
    def __init__(self, shape, layers=HIDDEN_LAYERS, activation=ACTIVATION):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.InputLayer(shape=(shape,)))
        for idx, neurons in enumerate(layers):
            self.model.add(
                keras.layers.Dense(
                    neurons,
                    activation=activation,
                    name=f'camada_oculta_{idx+1}_{activation}'
                )
            )
        self.model.add(keras.layers.Dense(1, activation='sigmoid', name='camada_saida_sigmoid'))
        self.compile()

    def compile(self, loss=LOSS, learning_rate=LEARNING_RATE):
        self.model.compile(
            loss=loss,      # Ideal para classificação binária
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate), # Otimizador popular e eficiente
            metrics=['accuracy']             # Métrica para acompanhar o desempenho
        )

    def summary(self):
        self.model.summary()

    def train(
        self,
        dataset,
        validation_size = 0.2,
        epochs=20,
        verbose=0
    ):
        dataset.split(validation_size),
        history = self.model.fit(
            dataset.features_train,
            dataset.target_train,
            epochs=epochs, # Número de vezes que o modelo verá todo o dataset
            validation_data=(dataset.features_test, dataset.target_test),
            verbose=verbose # Usamos verbose=0 para não poluir a saída, mas você pode usar 1 para ver o progresso
        )
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        print("\n--- Comparação Final Treino vs Validação ---")
        print(f"Acurácia final - Treino: {train_acc:.2%} | Validação: {val_acc:.2%}")
        print(f"Perda final    - Treino: {train_loss:.4f} | Validação: {val_loss:.4f}")
        if train_acc - val_acc > 0.05:
            print("\nPossível overfitting detectado: a acurácia de treino está significativamente maior que a de validação.")
        elif val_loss > train_loss * 1.2:
            print("\nPossível overfitting detectado: a perda de validação está significativamente maior que a de treino.")
        else:
            print("\nNão há sinais claros de overfitting.")
        
        return history.history

    def predict(self, dataset):
        loss, accuracy = self.model.evaluate(dataset.features_test, dataset.target_test, verbose=0)

        # --- Métricas detalhadas ---
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        target_predicted = (self.model.predict(dataset.features_test) > 0.5).astype(int)
        print("\nMétricas detalhadas no conjunto de validação:")
        print(f"Acurácia : {accuracy_score(dataset.target_test, target_predicted):.2%}")
        print(f"Precisão : {precision_score(dataset.target_test, target_predicted, zero_division=0):.2%}")
        print(f"Recall   : {recall_score(dataset.target_test, target_predicted, zero_division=0):.2%}")
        print(f"F1-score : {f1_score(dataset.target_test, target_predicted, zero_division=0):.2%}")

