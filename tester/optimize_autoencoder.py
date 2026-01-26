import os
import optuna
from optuna.pruners import MedianPruner
from dataset import Dataset
from autoencoder import Autoencoder
from mlp_modified import MLP
from sklearn.metrics import f1_score
import numpy as np


def objective(trial, dataset_path, target_column='CKD progression'):
    dataset = Dataset(dataset_path, target_column)
    
    n_layers = trial.suggest_int('n_layers', 2, 4)  # mínimo 2 camadas para ter bottleneck
    
    hidden_units = []
    input_size = dataset.get_shape()
    
    # garantir redução progressiva até o bottleneck
    for i in range(n_layers):
        if i == 0:
            min_units = max(16, input_size // 2)
            max_units = min(256, input_size * 2)
        else:
            min_units = 8
            max_units = max(16, hidden_units[-1] // 2)  # maximo 50% da camada anterior
            
            if max_units >= hidden_units[-1]:
                max_units = hidden_units[-1] - 8
        
        # garantir que min <= max
        if max_units < min_units:
            max_units = min_units
        
        step = 8 if max_units < 64 else 16
        
        units = trial.suggest_int(f'units_layer_{i}', min_units, max_units, step=step)
        hidden_units.append(units)
    
    mask_ratio = trial.suggest_float('mask_ratio', 0.1, 0.5)
    
    pretrain_lr = trial.suggest_float('pretrain_lr', 1e-4, 1e-2, log=True)
    
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    
    pretrain_epochs = trial.suggest_int('pretrain_epochs', 50, 300, step=50)
    
    categorical_indices = [2, 5, 6]
    categorical_cardinalities = {
        2: 4,  # etiology
        5: 4,  # CKD_stage
        6: 2   # proteinuria
    }
    
    try:
        autoencoder = Autoencoder(
            shape=dataset.get_shape(),
            categorical_indices=categorical_indices,
            categorical_cardinalities=categorical_cardinalities,
            mask_ratio=mask_ratio,
            hidden_units=hidden_units,
            learning_rate=pretrain_lr,
            mask_value=-999.0
        )
        
        history = autoencoder.train(
            dataset,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        val_loss = min(history['val_loss'])
        
        trial.report(val_loss, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return val_loss
    
    except Exception as e:
        print(f"Erro no trial {trial.number}: {e}")
        return float('inf')  


def optimize_hyperparameters(dataset_path, target_column='CKD progression', n_trials=50, study_name='autoencoder_optimization'):
    # criar o estudo (minimizar loss de reconstrução)
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, dataset_path, target_column),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    
    print("OTIMIZAÇÃO CONCLUÍDA")
    
    print(f"Número de trials concluídos: {len(study.trials)}\n")
    print(f"Melhor loss de validação: {study.best_value:.4f}")
    
    print("Melhores hiperparâmetros:\n")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    import joblib
    joblib.dump(study, f'{study_name}.pkl')
    print(f"Estudo salvo em: {study_name}.pkl\n")
    
    import matplotlib.pyplot as plt
    
    # histórico de otimização
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig('optuna_history.png', dpi=300, bbox_inches='tight')
    
    # importância dos hiperparâmetros
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig('optuna_importance.png', dpi=300, bbox_inches='tight')
    
    return study


def train_with_best_params(study, dataset_path, target_column='CKD progression'):
    """Treina o autoencoder com os melhores hiperparâmetros encontrados."""
    best_params = study.best_params
    
    dataset = Dataset(dataset_path, target_column)
    
    n_layers = best_params['n_layers']
    hidden_units = [best_params[f'units_layer_{i}'] for i in range(n_layers)]
    
    categorical_indices = [2, 5, 6]
    categorical_cardinalities = {2: 4, 5: 4, 6: 2}
    
    autoencoder = Autoencoder(
        shape=dataset.get_shape(),
        categorical_indices=categorical_indices,
        categorical_cardinalities=categorical_cardinalities,
        mask_ratio=best_params['mask_ratio'],
        hidden_units=hidden_units,
        learning_rate=best_params['pretrain_lr'],
        mask_value=-999.0
    )
    
    autoencoder.train(
        dataset,
        epochs=best_params['pretrain_epochs'],
        batch_size=best_params['batch_size'],
        verbose=1
    )
    
    autoencoder.save_encoder('best_pretrained_encoder.keras')
    print("\nAutoencoder otimizado salvo em: best_pretrained_encoder.keras")
    
    return autoencoder


if __name__ == "__main__":
    study = optimize_hyperparameters(
        dataset_path='datasets/dataset_filled_boruta_age_adults.csv',
        target_column='CKD progression',
        n_trials=20,  
        study_name='autoencoder_optimization'
    )
    
    best_autoencoder = train_with_best_params(
        study,
        dataset_path='datasets/dataset_filled_boruta_age_adults.csv',
        target_column='CKD progression'
    )
