import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, brier_score_loss
import optuna
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve


from multitask_mlp import MultiTaskMLP, HIDDEN_LAYERS, DROPOUTS, LEARNING_RATE, WEIGHT_DECAY, compute_ece
from multitask_dataset import MultiTaskDataset

SOURCE_PATH = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\age\dataset_idosos.csv"
TARGET_PATH = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\age\dataset_adultos.csv"

N_TRIALS = 100
EPOCHS_PRETRAIN = 200
EPOCHS_FINE_TUNE = 200
EPOCHS_FINAL = 200
BATCH_SIZE = 32
N_RUNS_FINAL = 40


def sample_hidden_layers(
    trial,
    *,
    max_layers=5,
    first_layer_choices=None,
    min_neurons=16
):
    if first_layer_choices is None:
        first_layer_choices = [32, 64, 128]

    n_layers = trial.suggest_int("n_layers", 2, max_layers)
    layer_1 = trial.suggest_categorical("layer_1", first_layer_choices)

    layers = [int(layer_1)]
    prev = layers[0]

    for i in range(2, n_layers + 1):
        ratio = trial.suggest_categorical(f"layer_{i}_ratio", ["same", "half"])
        if prev <= min_neurons:
            ratio = "same"

        nxt = prev if ratio == "same" else prev // 2
        if nxt < min_neurons:
            break

        layers.append(int(nxt))
        prev = nxt

    return layers


def objective(trial, source_dataset, target_dataset):
    hidden_layers = sample_hidden_layers(trial)
    trial.set_user_attr("hidden_layers", hidden_layers)
    trial.set_user_attr("n_layers", len(hidden_layers))

    n_layers = len(hidden_layers)

    dropouts = [
        trial.suggest_float(f"dropout_l{i+1}", 0.1, 0.5)
        for i in range(n_layers)
    ]

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

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

    best_t, best_f1 = model.find_best_threshold(y_val, val_probs)

    trial.set_user_attr("best_threshold", best_t)

    y_pred = (val_probs >= best_t).astype(int)

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )

    return macro_f1


if __name__ == "__main__":
    source_dataset = MultiTaskDataset(SOURCE_PATH)
    target_dataset = MultiTaskDataset(TARGET_PATH)
    print("Source dataset:", source_dataset)
    print("Target dataset:", target_dataset)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(
        trial, source_dataset, target_dataset), n_trials=N_TRIALS)

    print("===================================================")
    print("Melhores parâmetros encontrados pelo Optuna:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    best_threshold = study.best_trial.user_attrs.get("best_threshold", 0.5)
    print(f"Melhor F1 no conjunto de validação: {study.best_value:.4f}")
    print(f"Melhor threshold encontrado: {best_threshold:.3f}")
    print("===================================================")

    best_params = study.best_params
    HIDDEN_LAYERS[:] = study.best_trial.user_attrs["hidden_layers"]
    n_layers = len(HIDDEN_LAYERS)

    DROPOUTS[:] = [
        best_params[f"dropout_l{i+1}"]
        for i in range(n_layers)
    ]
    LEARNING_RATE = best_params["learning_rate"]
    WEIGHT_DECAY = best_params["weight_decay"]

    metrics_baseline = []
    metrics_finetuned = []

    for run in range(N_RUNS_FINAL):
        print(f"[INFO] Run final {run+1}/{N_RUNS_FINAL}")

        # Baseline: treinar apenas no target
        model_base = MultiTaskMLP(shape=target_dataset.features_train.shape[1])
        model_base.train(target_dataset, epochs=EPOCHS_FINAL, batch_size=BATCH_SIZE, verbose=0,
                         name=f"baseline_run{run}")

        val_outputs = model_base._predict_step(
            target_dataset.features_validation)
        val_probs = tf.reshape(val_outputs["prog"], [-1]).numpy()
        y_val = np.asarray(target_dataset.target_validation).ravel()
        best_t, _ = model_base.find_best_threshold(
            y_val, val_probs)

        test_outputs = model_base._predict_step(target_dataset.features_test)
        test_probs = tf.reshape(test_outputs["prog"], [-1]).numpy()
        y_true = np.asarray(target_dataset.target_test).ravel()
        y_pred = (test_probs >= best_t).astype(int)

        metrics_baseline.append({
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_class_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
            "f1_class_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "auc_roc": roc_auc_score(y_true, test_probs),
            "brier": brier_score_loss(y_true, test_probs),
            "ece": compute_ece(y_true, test_probs)
        })

        # Fine-tuned target: pré-treino no source + fine-tune
        model_ft = MultiTaskMLP(shape=target_dataset.features_train.shape[1])
        model_ft.train(source_dataset, epochs=EPOCHS_PRETRAIN, batch_size=BATCH_SIZE, verbose=0,
                       name=f"finetuned_pretrain_run{run}")
        model_ft.train(target_dataset, epochs=EPOCHS_FINAL, batch_size=BATCH_SIZE, verbose=0,
                       name=f"finetuned_finetune_run{run}")

        val_outputs = model_ft._predict_step(
            target_dataset.features_validation)
        val_probs = tf.reshape(val_outputs["prog"], [-1]).numpy()
        best_t, _ = model_ft.find_best_threshold(
            y_val, val_probs)

        test_outputs = model_ft._predict_step(target_dataset.features_test)
        test_probs = tf.reshape(test_outputs["prog"], [-1]).numpy()
        y_pred = (test_probs >= best_t).astype(int)

        metrics_finetuned.append({
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_class_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
            "f1_class_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "auc_roc": roc_auc_score(y_true, test_probs),
            "brier": brier_score_loss(y_true, test_probs),
            "ece": compute_ece(y_true, test_probs)
        })

    # calcular médias
    avg_baseline = {k: np.mean([m[k] for m in metrics_baseline])
                    for k in metrics_baseline[0]}
    avg_finetuned = {k: np.mean([m[k] for m in metrics_finetuned])
                     for k in metrics_finetuned[0]}

    print("===================================================")
    print("Resultados médios finais (10 runs):")
    print("scenario,accuracy,precision,recall,f1_score,auc_roc,brier,ece")
    print(f"baseline,{avg_baseline['accuracy']:.4f},{avg_baseline['precision_macro']:.4f},"
          f"{avg_baseline['recall_macro']:.4f},{avg_baseline['f1_macro']:.4f},{avg_baseline['auc_roc']:.4f},"
          f"{avg_baseline['brier']:.4f},{avg_baseline['ece']:.4f}")
    print(f"fine_tuned_target,{avg_finetuned['accuracy']:.4f},{avg_finetuned['precision_macro']:.4f},"
          f"{avg_finetuned['recall_macro']:.4f},{avg_finetuned['f1_macro']:.4f},{avg_finetuned['auc_roc']:.4f},"
          f"{avg_finetuned['brier']:.4f},{avg_finetuned['ece']:.4f}")
    print("===================================================")

    csv_file = "mlp_results_avg.csv"
    fieldnames = [
        "scenario",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_class_0",
        "f1_class_1",
        "auc_roc",
        "brier",
        "ece"
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"scenario": "baseline", **avg_baseline})
        writer.writerow({"scenario": "fine_tuned_target", **avg_finetuned})

    print(f"[INFO] Resultados médios finais salvos em {csv_file}")

    plt.figure(figsize=(8, 6))

    test_probs_base = np.reshape(model_base._predict_step(
        target_dataset.features_test)["prog"], [-1])
    y_pred_base = (test_probs_base >= best_threshold).astype(int)

    test_probs_ft = np.reshape(model_ft._predict_step(
        target_dataset.features_test)["prog"], [-1])
    y_pred_ft = (test_probs_ft >= best_threshold).astype(int)

    plt.plot([0, 1], [0, 1], 'k--', label="Random")

    fpr_base, tpr_base, _ = roc_curve(y_true, test_probs_base)
    roc_auc_base = auc(fpr_base, tpr_base)
    plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC = {roc_auc_base:.2f})')

    fpr_ft, tpr_ft, _ = roc_curve(y_true, test_probs_ft)
    roc_auc_ft = auc(fpr_ft, tpr_ft)
    plt.plot(fpr_ft, tpr_ft, label=f'Fine-tuned (AUC = {roc_auc_ft:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # Salvar gráfico
    roc_file = "roc_curve.png"
    plt.savefig(roc_file, dpi=300)
    plt.close()
    print(f"[INFO] Curva ROC salva em {roc_file}")

    plt.figure(figsize=(8, 6))

    prob_true_base, prob_pred_base = calibration_curve(
        y_true, test_probs_base, n_bins=15, strategy="quantile"
    )

    prob_true_ft, prob_pred_ft = calibration_curve(
        y_true, test_probs_ft, n_bins=15, strategy="quantile"
    )

    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    plt.plot(
        prob_pred_base,
        prob_true_base,
        marker="o",
        label=f"Baseline (Brier = {avg_baseline['brier']:.3f})"
    )

    plt.plot(
        prob_pred_ft,
        prob_true_ft,
        marker="o",
        label=f"Fine-tuned (Brier = {avg_finetuned['brier']:.3f})"
    )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Brier Score)")
    plt.legend(loc="upper left")
    plt.grid(True)

    calibration_file = "brier_calibration_curve.png"
    plt.savefig(calibration_file, dpi=300)
    plt.close()

    print(f"[INFO] Curva de calibração (Brier) salva em {calibration_file}")
