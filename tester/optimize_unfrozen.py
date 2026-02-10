import os
import optuna
from optuna.pruners import MedianPruner
from dataset import Dataset
from mlp_modified import MLP
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from tensorflow import keras
import numpy as np


def objective(trial, dataset_path, encoder_path, target_column='CKD progression'):
    dataset = Dataset(dataset_path, target_column)
    
    pretrained_encoder = keras.models.load_model(encoder_path)
    
    finetune_lr = trial.suggest_float('finetune_lr', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    finetune_epochs = trial.suggest_int('finetune_epochs', 50, 300, step=50)
    
    try:
        mlp = MLP(
            shape=dataset.get_shape(),
            pretrained_encoder=pretrained_encoder,
            freeze_encoder=True
        )
        
        mlp.unfreeze_encoder(learning_rate=finetune_lr)
        
        mlp.train(
            dataset,
            epochs=finetune_epochs,
            batch_size=batch_size,
            verbose=0,
            plot_path=None
        )
        
        y_pred = mlp.model.predict(dataset.features_validation, verbose=0)
        y_pred_class = (y_pred > 0.5).astype(int).flatten()
        y_true = dataset.target_validation.values

        import numpy as np
        print(f"Distribuição das predições: {np.unique(y_pred_class, return_counts=True)}")
        print(f"Distribuição dos targets: {np.unique(y_true, return_counts=True)}")
        recall = recall_score(y_true, y_pred_class)
        print(f"Recall: {recall:.4f}")
        
        f1 = f1_score(y_true, y_pred_class)
        print(f"F1-score: {f1:.4f}")
        
        trial.report(f1, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return f1
    
    except Exception as e:
        print(f"Erro no trial {trial.number}: {e}")
        return 0.0


def optimize_hyperparameters(
    dataset_path,
    encoder_path='best_pretrained_encoder.keras',
    target_column='CKD progression',
    n_trials=50,
    study_name='unfrozen_optimization'
):
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, dataset_path, encoder_path, target_column),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("OTIMIZAÇÃO CONCLUÍDA")
    print(f"Número de trials concluídos: {len(study.trials)}\n")
    print(f"Melhor F1-score: {study.best_value:.4f}")
    
    print("Melhores hiperparâmetros:\n")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    import joblib
    joblib.dump(study, f'{study_name}.pkl')
    print(f"Estudo salvo em: {study_name}.pkl\n")
    
    import matplotlib.pyplot as plt
    
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig('plot/optuna_unfrozen_history.png', dpi=300, bbox_inches='tight')
    
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig('plot/optuna_unfrozen_importance.png', dpi=300, bbox_inches='tight')
    
    return study


def train_with_best_params(
    study,
    dataset,
    encoder_path='best_pretrained_encoder.keras',
):
    best_params = study.best_params
    
    pretrained_encoder = keras.models.load_model(encoder_path)
    
    mlp = MLP(
        shape=dataset.get_shape(),
        pretrained_encoder=pretrained_encoder,
        freeze_encoder=False
    )
    
    mlp.unfreeze_encoder(learning_rate=best_params['finetune_lr'])
    
    mlp.train(
        dataset,
        epochs=best_params['finetune_epochs'],
        batch_size=best_params['batch_size'],
        verbose=1,
        plot_path="best_unfrozen"
    )
    
    return mlp


if __name__ == "__main__":
    study = optimize_hyperparameters(
        dataset_path='datasets/dataset_filled_boruta_age_adults.csv',
        encoder_path='best_pretrained_encoder.keras',
        target_column='CKD progression',
        n_trials=20,
        study_name='unfrozen_optimization'
    )

    dataset = Dataset('datasets/dataset_filled_boruta_age_adults.csv', 'CKD progression')

    # import joblib
    # study = joblib.load('unfrozen_optimization.pkl')
    
    best_model = train_with_best_params(
        study,
        dataset,
        encoder_path='best_pretrained_encoder.keras',
    )

    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = best_model.model.predict(dataset.features_test)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()
    y_true = dataset.target_test.values
    print("RELATÓRIO DE CLASSIFICAÇÃO DO MELHOR MODELO OTIMIZADO:\n")
    print(classification_report(y_true, y_pred_class, digits=4))
    print("MATRIZ DE CONFUSÃO DO MELHOR MODELO OTIMIZADO:\n")
    print(confusion_matrix(y_true, y_pred_class))
    print("CURVA ROC AUC DO MELHOR MODELO OTIMIZADO:\n")
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    print(f"AUC-ROC: {auc_score:.4f}")
