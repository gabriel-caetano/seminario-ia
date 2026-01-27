import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dataset import Dataset
from autoencoder import Autoencoder
from mlp_modified import MLP
from tensorflow import keras
from sklearn.metrics import classification_report


def pretrain_and_finetune(dataset_path, target_column='CKD progression'):
    dataset = Dataset(dataset_path, target_column)
    
    print("FASE 1: Pré-treino Auto-supervisionado\n")
    
    # index: 0=age, 1=SBP, 2=etiology, 3=Hb, 4=Alb, , 5=CKD_stage, 6=proteinuria, 7=UPCR
    categorical_indices = [2, 5, 6]  # etiology, CKD_stage, proteinuria
    categorical_cardinalities = {
        2: 4,  # etiology: [1, 2, 3, 4]
        5: 4,  # CKD_stage: [2, 3, 4, 5]
        6: 2   # proteinuria: [0, 1]
    }
    
    reconstruction_model = Autoencoder(
        shape=dataset.get_shape(),
        categorical_indices=categorical_indices,
        categorical_cardinalities=categorical_cardinalities,
        mask_ratio=0.3,
        hidden_units=[128, 64],
        learning_rate=0.001,
        mask_value=-999.0
    )
    
    reconstruction_model.train(
        dataset,
        epochs=200,
        batch_size=16,
        verbose=1
    )
    
    reconstruction_model.save_encoder('pretrained_encoder.keras')
    
    print("FASE 2: Fine-tuning Supervisionado\n")
    
    pretrained_encoder = reconstruction_model.get_encoder()
    # mlp_pretrained = MLP(
    #     shape=dataset.get_shape(),
    #     pretrained_encoder=pretrained_encoder
    # )

    # pretrained_encoder = keras.models.load_model('pretrained_encoder.keras')

    mlp_pretrained_frozen = MLP(
        shape=dataset.get_shape(),
        pretrained_encoder=pretrained_encoder,
        freeze_encoder=True
    )

    mlp_pretrained_unfrozen = MLP(
        shape=dataset.get_shape(),
        pretrained_encoder=pretrained_encoder,
        freeze_encoder=False
    )

    
    print("Fase 2a: Fine-tune com encoder congelado\n")
    mlp_pretrained_frozen.train(
        dataset,
        epochs=200,
        batch_size=16,
        verbose=1,
        plot_path="finetune_frozen"
    )
    
    print("Fase 2b: Fine-tune com encoder descongelado\n")
    mlp_pretrained_unfrozen.unfreeze_encoder(learning_rate=0.0001)
    mlp_pretrained_unfrozen.train(
        dataset,
        epochs=200,
        batch_size=16,
        verbose=1,
        plot_path="finetune_unfrozen"
    )
    
    
    print("RESULTADOS")

    y_pred_frozen = mlp_pretrained_frozen.model.predict(dataset.features_test)
    y_true = dataset.target_test

    print("-"*80)
    print("RESULTADOS FROZEN\n")
    print(classification_report(y_true, y_pred_frozen.round(), digits=4))

    results_frozen = classification_report(y_true, y_pred_frozen.round(), output_dict=True)

    y_pred_unfrozen = mlp_pretrained_unfrozen.model.predict(dataset.features_test)

    print("-"*80)
    print("RESULTADOS UNFROZEN\n")
    print(classification_report(y_true, y_pred_unfrozen.round(), digits=4))

    results_unfrozen = classification_report(y_true, y_pred_unfrozen.round(), output_dict=True)

    # print("-"*80)
    # print("RESULTADOS UNFROZEN\n")
    # for metric, value in results_unfrozen.items():
    #     print(f"{metric}: {value:.4f}")

    # print("-"*80)
    # print("RESULTADOS FROZEN\n")
    # for metric, value in results_frozen.items():
    #     print(f"{metric}: {value:.4f}")
    
    print("FASE 3: Modelo do Zero (Baseline)\n")
    
    mlp_baseline = MLP(shape=dataset.get_shape())
    mlp_baseline.train(
        dataset,
        epochs=200,
        batch_size=16,
        verbose=1,
        plot_path="baseline"
    )
    
    y_pred_baseline = mlp_baseline.model.predict(dataset.features_test)
    print("-"*80)
    print("RESULTADOS BASELINE\n")
    print(classification_report(y_true, y_pred_baseline.round(), digits=4))
    print("-"*80)

    results_baseline = classification_report(y_true, y_pred_baseline.round(), output_dict=True)

    
    print("COMPARAÇÃO: PRÉ-TREINO vs BASELINE\n")
    print(f"{'Métrica':<20} {'Frozen':>12} {'Unfrozen':>12} {'Baseline':>12} {'Δ':>10}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1-score']:
        if metric == 'accuracy':
            key = 'accuracy'
        else:
            key = 'macro avg'
        pretrain_frozen_value = results_frozen[key][metric] if metric != 'accuracy' else results_frozen[key]
        pretrain_unfrozen_value = results_unfrozen[key][metric] if metric != 'accuracy' else results_unfrozen[key]
        baseline_value = results_baseline[key][metric] if metric != 'accuracy' else results_baseline[key]
        delta_frozen = pretrain_frozen_value - baseline_value
        delta_unfrozen = pretrain_unfrozen_value - baseline_value
        print(f"{metric:<20} {pretrain_frozen_value:>12.4f} {pretrain_unfrozen_value:>12.4f} {baseline_value:>12.4f} {delta_frozen:>+10.4f} / {delta_unfrozen:>+10.4f}")

if __name__ == "__main__":
    pretrain_and_finetune(
        'datasets/dataset_filled_boruta_age_adults.csv',
        'CKD progression'
    )