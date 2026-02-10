import os
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)

HIDDEN_LAYERS = [128, 128, 128, 128, 128]
DROPOUTS = [0.2, 0.2, 0.2, 0.2, 0.2]
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.0
ACTIVATION = "relu"
THRESHOLD_STEPS = 100
PROTEINURIA_LOSS_WEIGHT = 0.2

def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += abs(bin_conf - bin_acc) * mask.sum() / len(y_true)
    return ece

class MultiTaskMLP:
    def __init__(self, shape):
        inputs = keras.layers.Input(shape=(shape,))
        x = inputs

        for idx, neurons in enumerate(HIDDEN_LAYERS):
            x = keras.layers.Dense(
                neurons,
                activation=ACTIVATION,
                kernel_regularizer=regularizers.l2(1e-4),
                name=f"hidden_{idx+1}",
            )(x)
            x = keras.layers.Dropout(DROPOUTS[idx], name=f"dropout_{idx+1}")(x)

        prog_output = keras.layers.Dense(1, activation="sigmoid", name="prog")(x)
        proteinuria_output = keras.layers.Dense(1, activation="sigmoid", name="proteinuria")(x)

        self.model = keras.Model(inputs=inputs, outputs={"prog": prog_output, "proteinuria": proteinuria_output})
        self.temperature = tf.Variable(1.0, dtype=tf.float32, trainable=True)
        self.history = None
        self.compile()

    def summary(self):
        self.model.summary()

    def compile(self):
        self.model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
            loss={"prog": "binary_crossentropy", "proteinuria": "binary_crossentropy"},
            loss_weights={"prog": 1.0, "proteinuria": PROTEINURIA_LOSS_WEIGHT},
            metrics={"prog": [keras.metrics.AUC(name="auc")], "proteinuria": [keras.metrics.AUC(name="auc")]}
        )

    def train(self, dataset, epochs=2000, batch_size=32, verbose=1, name="run"):
        os.makedirs("logs", exist_ok=True)
        early_stopping = EarlyStopping(monitor="val_prog_auc", mode="max", patience=15, restore_best_weights=True)

        history = self.model.fit(
            dataset.features_train,
            {"prog": dataset.target_train, "proteinuria": dataset.proteinuria_train},
            validation_data=(dataset.features_validation, {"prog": dataset.target_validation, "proteinuria": dataset.proteinuria_validation}),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose,
        )
        self.history = history.history
        return history.history

    @staticmethod
    def probs_to_logits_tf(probs, eps=1e-6):
        probs = tf.clip_by_value(probs, eps, 1.0 - eps)
        return tf.math.log(probs / (1.0 - probs))

    def fit_temperature(self, probs, y_true, lr=0.01, epochs=300):
        probs = tf.convert_to_tensor(probs, dtype=tf.float32)
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        logits = self.probs_to_logits_tf(probs)
        optimizer = tf.keras.optimizers.Adam(lr)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                scaled_logits = logits / self.temperature
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, tf.sigmoid(scaled_logits)))
            grads = tape.gradient(loss, [self.temperature])
            optimizer.apply_gradients(zip(grads, [self.temperature]))
        print(f"[INFO] Melhor temperature: {self.temperature.numpy():.3f}")

    def find_best_threshold(self, y_true, y_prob, min_recall_target=0.5):
        thresholds = np.linspace(0.05, 0.95, THRESHOLD_STEPS)
        best_score = -1  
        best_t = 0.5
        best_f1 = 0.0

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            r = recall_score(y_true, y_pred, zero_division=0)
            p = precision_score(y_true, y_pred, zero_division=0)

            if r < min_recall_target:
                continue

            score = f1_score(y_true, y_pred, zero_division=0) - 0.1 * abs(p - r)

            if score > best_score:
                best_score = score
                best_t = t
                best_f1 = f1_score(y_true, y_pred, zero_division=0)

        return best_t, best_f1

    @tf.function(reduce_retracing=True)
    def _predict_step(self, inputs):
        return self.model(inputs, training=False)

    def predict_all(self, dataset, min_recall_target=0.5):
        val_outputs = self._predict_step(dataset.features_validation)
        val_probs = tf.reshape(val_outputs["prog"], [-1])
        y_val = tf.convert_to_tensor(np.asarray(dataset.target_validation).ravel(), dtype=tf.float32)

        self.fit_temperature(val_probs, y_val)
        val_probs_cal = tf.sigmoid(self.probs_to_logits_tf(val_probs) / self.temperature).numpy()

        best_t, best_f1 = self.find_best_threshold(y_val.numpy(), val_probs_cal, min_recall_target=min_recall_target)
        print(f"[INFO] Melhor threshold: {best_t:.3f} | F1: {best_f1:.3f}")

        test_outputs = self._predict_step(dataset.features_test)
        test_probs = tf.reshape(test_outputs["prog"], [-1])
        test_probs_cal = tf.sigmoid(self.probs_to_logits_tf(test_probs) / self.temperature).numpy()
        y_pred = (test_probs_cal >= best_t).astype(int)
        y_true = np.asarray(dataset.target_test).ravel()
        auc = float(tf.keras.metrics.AUC()(y_true, test_probs_cal).numpy())

        results = {
            "prog": {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "auc_roc": auc,
                "brier": brier_score_loss(y_true, test_probs_cal),
                "ece": compute_ece(y_true, test_probs_cal),
            },
            "debug": {
                "temperature": float(self.temperature.numpy()),
                "best_threshold": float(best_t),
                "best_f1_val": float(best_f1),
            },
        }

        self.save_log_geral(dataset, results)
        return results

    def save_log_geral(self, dataset, results, filename="log_geral.json"):
        log = {
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "train_size": int(len(dataset.target_train)),
                "val_size": int(len(dataset.target_validation)),
                "test_size": int(len(dataset.target_test)),
            },
            "model_config": {
                "hidden_layers": HIDDEN_LAYERS,
                "dropouts": DROPOUTS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "activation": ACTIVATION,
            },
            "training": {},
            "calibration": results["debug"],
            "test_metrics": results["prog"],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, ensure_ascii=False)

        print(f"[INFO] log salvo em {filename}")
