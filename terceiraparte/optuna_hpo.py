import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
import optuna

from multitask_mlp import MultiTaskMLP, HIDDEN_LAYERS, DROPOUTS, LEARNING_RATE, WEIGHT_DECAY
from multitask_dataset import MultiTaskDataset

SOURCE_PATH = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\age\dataset_idosos.csv"
TARGET_PATH = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\age\dataset_adultos.csv"

N_TRIALS = 20
EPOCHS_PRETRAIN = 50
EPOCHS_FINE_TUNE = 50
EPOCHS_FINAL = 200
BATCH_SIZE = 32

def objective(trial, source_dataset, target_dataset):
    # Sugestão de hiperparâmetros pelo Optuna
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden_layers = [trial.suggest_int(f"n_neurons_l{i+1}", 64, 256) for i in range(n_layers)]
    dropouts = [trial.suggest_float(f"dropout_l{i+1}", 0.1, 0.5) for i in range(n_layers)]
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    min_recall_target = trial.suggest_float("min_recall_target", 0.3, 0.8)

    # Atualiza os globais do MultiTaskMLP
    global HIDDEN_LAYERS, DROPOUTS, LEARNING_RATE, WEIGHT_DECAY
    HIDDEN_LAYERS[:] = hidden_layers
    DROPOUTS[:] = dropouts
    LEARNING_RATE = learning_rate
    WEIGHT_DECAY = weight_decay

    model = MultiTaskMLP(shape=source_dataset.features_train.shape[1])

    model.train(source_dataset, epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE, verbose=0,
                name=f"trial_{trial.number}_pretrain")

    model.train(target_dataset, epochs=EPOCHS_FINE_TUNE, batch_size=BATCH_SIZE, verbose=0,
                name=f"trial_{trial.number}_finetune")

    val_outputs = model._predict_step(target_dataset.features_validation)
    val_probs = tf.reshape(val_outputs["prog"], [-1]).numpy()
    y_val = np.asarray(target_dataset.target_validation).ravel()

    best_t, best_f1 = model.find_best_threshold(y_val, val_probs, min_recall_target=min_recall_target)

    trial.set_user_attr("best_threshold", best_t)
    trial.set_user_attr("min_recall_target", min_recall_target)

    return best_f1


if __name__ == "__main__":
    source_dataset = MultiTaskDataset(SOURCE_PATH)
    target_dataset = MultiTaskDataset(TARGET_PATH)
    print("Source dataset:", source_dataset)
    print("Target dataset:", target_dataset)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, source_dataset, target_dataset), n_trials=N_TRIALS)

    print("===================================================")
    print("Melhores parâmetros encontrados pelo Optuna:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    best_threshold = study.best_trial.user_attrs.get("best_threshold", 0.5)
    min_recall_target = study.best_trial.user_attrs.get("min_recall_target", 0.5)

    print(f"Melhor F1 no conjunto de validação do target: {study.best_value:.4f}")
    print(f"Melhor threshold encontrado: {best_threshold:.3f}")
    print(f"Min recall target usado: {min_recall_target:.2f}")
    print("===================================================")

    best_params = study.best_params
    n_layers = best_params.get("n_layers", 3)
    HIDDEN_LAYERS[:] = [best_params[f"n_neurons_l{i+1}"] for i in range(n_layers)]
    DROPOUTS[:] = [best_params[f"dropout_l{i+1}"] for i in range(n_layers)]
    LEARNING_RATE = best_params["learning_rate"]
    WEIGHT_DECAY = best_params["weight_decay"]

    final_model = MultiTaskMLP(shape=target_dataset.features_train.shape[1])

    final_model.train(source_dataset, epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE, verbose=1, name="final_pretrain")
    final_model.train(target_dataset, epochs=EPOCHS_FINAL, batch_size=BATCH_SIZE, verbose=1, name="final_finetune")

    val_outputs = final_model._predict_step(target_dataset.features_validation)
    val_probs = tf.reshape(val_outputs["prog"], [-1]).numpy()
    y_val = np.asarray(target_dataset.target_validation).ravel()
    best_t, best_f1 = final_model.find_best_threshold(y_val, val_probs, min_recall_target=min_recall_target)

    print(f"Threshold final usado no modelo: {best_t:.3f}")
    print(f"F1 no conjunto de validação do target (condicionado ao recall mínimo): {best_f1:.4f}")
    print("[INFO] Treinamento final concluído!")
