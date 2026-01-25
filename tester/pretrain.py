import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dataset import Dataset
from reconstruction_mlp import ReconstructionMLP
from mlp_modified import MLP

def pretrain_and_finetune(dataset_path, target_column='CKD progression'):
    dataset = Dataset(dataset_path, target_column)
    
    print("FASE 1: Pré-treino Auto-supervisionado\n")
    
    # index: 0=age, 1=SBP, 2=etiology, 3=Hb, 4=Alb, 5=eGFR, 6=CKD_stage, 7=proteinuria, 8=UPCR
    categorical_indices = [2, 6, 7]  # etiology, CKD_stage, proteinuria
    categorical_cardinalities = {
        2: 4,  # etiology: [1, 2, 3, 4]
        6: 4,  # CKD_stage: [2, 3, 4, 5]
        7: 2   # proteinuria: [0, 1]
    }
    
    reconstruction_model = ReconstructionMLP(
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
    mlp_pretrained = MLP(
        shape=dataset.get_shape(),
        pretrained_encoder=pretrained_encoder
    )
    
    print("Fase 2a: Fine-tune com encoder congelado\n")
    print("-"*80)
    mlp_pretrained.train(
        dataset,
        epochs=50,
        batch_size=16,
        verbose=1,
        name="finetune_frozen"
    )

    mlp_frozen = mlp_pretrained
    
    print("Fase 2b: Fine-tune com encoder descongelado\n")
    print("-"*80)
    mlp_pretrained.unfreeze_encoder(learning_rate=0.0001)
    mlp_pretrained.train(
        dataset,
        epochs=50,
        batch_size=16,
        verbose=1,
        name="finetune_unfrozen"
    )
    
    mlp_unfrozen = mlp_pretrained
    
    print("RESULTADOS")
    
    results_unfrozen = mlp_unfrozen.predict(dataset)
    results_frozen = mlp_frozen.predict(dataset)
    
    print("FASE 3: Modelo do Zero (Baseline)\n")
    
    mlp_baseline = MLP(shape=dataset.get_shape())
    mlp_baseline.train(
        dataset,
        epochs=100,
        batch_size=16,
        verbose=0,
        name="baseline"
    )
    
    results_baseline = mlp_baseline.predict(dataset)
    
    print("COMPARAÇÃO: PRÉ-TREINO vs BASELINE\n")
    print(f"{'Métrica':<20} {'Pré-treino':>12} {'Pré-treino':>12} {'Baseline':>12} {'Δ':>10}")
    
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        unfrozen = results_unfrozen[metric]
        frozen = results_frozen[metric]
        baseline = results_baseline[metric]
        delta_unfrozen = unfrozen - baseline
        delta_frozen = frozen - baseline
        print(f"{metric:<20} {unfrozen:>12.4f} {frozen:>12.4f} {baseline:>12.4f} {delta_unfrozen:>+12.4f} {delta_frozen:>+12.4f}")

if __name__ == "__main__":
    pretrain_and_finetune(
        'datasets/dataset_filled_boruta_age_adults.csv',
        'CKD progression'
    )