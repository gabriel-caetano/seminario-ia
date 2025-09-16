from tensorflow import keras

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
        dataset_source,
        dataset_target=None,
        epochs=20,
        verbose=0
    ):
        if dataset_target is None:
            history = self.model.fit(
                dataset_source.features_train,
                dataset_source.target_train,
                epochs=epochs, # Número de vezes que o modelo verá todo o dataset
                validation_data=(dataset_source.features_test, dataset_source.target_test),
                verbose=verbose # Usamos verbose=0 para não poluir a saída, mas você pode usar 1 para ver o progresso
            )
        else:
            history = self.model.fit(
                dataset_target.features_train,
                dataset_target.target_train,
                epochs=epochs, # Número de vezes que o modelo verá todo o dataset
                validation_data=(dataset_target.features_test, dataset_target.target_test),
                verbose=verbose # Usamos verbose=0 para não poluir a saída, mas você pode usar 1 para ver o progresso
            )
        return history

        