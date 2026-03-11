import os
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
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

MIN_RECALL_TARGET = 0.6
THRESHOLD_STEPS = 100

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
            x = keras.layers.Dropout(
                DROPOUTS[idx],
                name=f"dropout_{idx+1}",
            )(x)

        outputs = keras.layers.Dense(
            1, activation="sigmoid", name="prog"
        )(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.temperature = tf.Variable(1.0, dtype=tf.float32, trainable=True)

        self.compile()
        self.history = None

    def summary(self):
        self.model.summary()

    def compile(self):
        self.model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
            ),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )

    def train(self, dataset, epochs=2000, batch_size=32, verbose=1, name="run"):
        os.makedirs("logs", exist_ok=True)
        os.makedirs("plot", exist_ok=True)

        early_stopping = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=15,
            restore_best_weights=True,
        )

        csv_logger = CSVLogger(
            f"logs/{name}_training_log.csv",
            append=False,
        )

        history = self.model.fit(
            dataset.features_train,
            dataset.target_train,
            validation_data=(
                dataset.features_validation,
                dataset.target_validation,
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, csv_logger],
            verbose=verbose,
        )

        self.history = history.history
        self.plot_training_curves(name)

        return history.history

    def plot_training_curves(self, filename="training"):
        if self.history is None:
            return

        plt.figure(figsize=(6, 4))
        plt.plot(self.history["auc"], label="Train AUC")
        plt.plot(self.history["val_auc"], label="Val AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plot/{filename}_auc.png", dpi=300)
        plt.close()

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
                loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        y_true, tf.sigmoid(scaled_logits)
                    )
                )
            grads = tape.gradient(loss, [self.temperature])
            optimizer.apply_gradients(zip(grads, [self.temperature]))

        print(f"[INFO] Temperature ótima: {self.temperature.numpy():.3f}")

    def find_best_threshold(self, y_true, y_prob):
        thresholds = np.linspace(0.05, 0.95, THRESHOLD_STEPS)
        best_t, best_f1 = 0.5, -1

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            r = recall_score(y_true, y_pred, zero_division=0)

            if r < MIN_RECALL_TARGET:
                continue

            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        return best_t, best_f1

    @tf.function(reduce_retracing=True)
    def _predict_step(self, inputs):
        return self.model(inputs, training=False)

    def predict_all(self, dataset):
        # ===== VAL =====
        val_probs = tf.reshape(
            self._predict_step(dataset.features_validation), [-1]
        )
        y_val = tf.convert_to_tensor(
            np.asarray(dataset.target_validation).ravel(),
            dtype=tf.float32,
        )

        self.fit_temperature(val_probs, y_val)

        val_probs_cal = tf.sigmoid(
            self.probs_to_logits_tf(val_probs) / self.temperature
        ).numpy()

        best_t, best_f1 = self.find_best_threshold(
            y_val.numpy(), val_probs_cal
        )

        print(
            f"[INFO] Threshold ótimo (val): {best_t:.3f} | F1: {best_f1:.3f}"
        )

        test_probs = tf.reshape(
            self._predict_step(dataset.features_test), [-1]
        )

        test_probs_cal = tf.sigmoid(
            self.probs_to_logits_tf(test_probs) / self.temperature
        ).numpy()

        y_pred = (test_probs_cal >= best_t).astype(int)
        y_true = np.asarray(dataset.target_test).ravel()

        auc = float(
            tf.keras.metrics.AUC()(y_true, test_probs_cal).numpy()
        )

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

        if self.history is not None:
            val_auc = self.history.get("val_auc", [])
            train_auc = self.history.get("auc", [])

            best_epoch = int(np.argmax(val_auc))

            log["training"] = {
                "epochs_ran": len(val_auc),
                "best_epoch": best_epoch,
                "best_val_auc": float(val_auc[best_epoch]),
                "final_val_auc": float(val_auc[-1]),
                "final_train_auc": float(train_auc[-1]),
                "overfit_gap": float(train_auc[-1] - val_auc[-1]),
            }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, ensure_ascii=False)

        print(f"[INFO] log_geral salvo em {filename}")
