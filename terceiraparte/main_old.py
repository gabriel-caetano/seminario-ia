import os
import csv
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from multitask_dataset import MultiTaskDataset
from multitask_mlp import MultiTaskMLP


class SingleTaskTransferTester:
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def run_baseline(self):
        model = MultiTaskMLP(shape=self.target_dataset.features_train.shape[1])
        model.summary()
        model.train(
            self.target_dataset,
            epochs=2000,
            batch_size=32,
            verbose=1,
            name='single_run_baseline'
        )
        return model.predict_all(self.target_dataset)

    def run_transfer(self):
        model = MultiTaskMLP(shape=self.source_dataset.features_train.shape[1])
        model.summary()

        model.train(
            self.source_dataset,
            epochs=5000,
            batch_size=32,
            verbose=1,
            name='single_run_pretrain_source'
        )
        source_metrics = model.predict_all(self.source_dataset)

        model.train(
            self.target_dataset,
            epochs=2000,
            batch_size=32,
            verbose=1,
            name='single_run_finetune_target'
        )
        fine_tuned_metrics = model.predict_all(self.target_dataset)

        return {
            'source_dataset': source_metrics,
            'fine_tuned_target': fine_tuned_metrics
        }

    def run(self):
        baseline = self.run_baseline()
        transfer = self.run_transfer()
        return {
            'baseline': baseline,
            'transfer_learning': transfer
        }

    def save_results_csv(self, results, directory="results"):
        os.makedirs(directory, exist_ok=True)
        task = 'prog'
        filename = os.path.join(directory, f"{task}_results.csv")

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['scenario','accuracy','precision','recall','f1_score','auc_roc','brier','ece'])
            writer.writeheader()

            # Baseline
            row = {'scenario': 'baseline'}
            row.update(results['baseline'][task])
            writer.writerow(row)

            # Source dataset (pré-treino)
            row = {'scenario': 'source_dataset'}
            row.update(results['transfer_learning']['source_dataset'][task])
            writer.writerow(row)

            # Fine-tuned target
            row = {'scenario': 'fine_tuned_target'}
            row.update(results['transfer_learning']['fine_tuned_target'][task])
            writer.writerow(row)

        print(f"{task.upper()} resultados salvos em: {filename}")

if __name__ == "__main__":
    print("\n--- Experimento Single Run Transfer Learning CKD Progression ---")

    dataset_source = MultiTaskDataset(
        file_name='datasets/stage/dataset_stage234.csv',
        target_prog='CKD progression'
    )

    dataset_target = MultiTaskDataset(
        file_name='datasets/stage/dataset_stage5.csv',
        target_prog='CKD progression'
    )

    tester = SingleTaskTransferTester(
        source_dataset=dataset_source,
        target_dataset=dataset_target
    )

    results = tester.run()

    print("\nResultados finais:")
    print(results)

    tester.save_results_csv(results)
