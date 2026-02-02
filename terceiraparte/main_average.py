import os
import csv
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from multitask_dataset import MultiTaskDataset
from multitask_mlp import MultiTaskMLP


N_RUNS = 10
TASK = 'prog'


class SingleTaskTransferTester:
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def run_baseline(self):
        model = MultiTaskMLP(
            shape=self.target_dataset.features_train.shape[1]
        )

        model.train(
            self.target_dataset,
            epochs=2000,
            batch_size=32,
            verbose=0,
            name='multitask_run_baseline'
        )

        return model.predict_all(self.target_dataset)

    def run_transfer(self):
        model = MultiTaskMLP(
            shape=self.source_dataset.features_train.shape[1]
        )

        model.train(
            self.source_dataset,
            epochs=5000,
            batch_size=32,
            verbose=0,
            name='multitask_run_pretrain_source'
        )

        source_metrics = model.predict_all(self.source_dataset)

        model.train(
            self.target_dataset,
            epochs=2000,
            batch_size=32,
            verbose=0,
            name='multitask_run_finetune_target'
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

def aggregate_metrics(results_list, scenario):
    metrics = {}
    for key in results_list[0][scenario][TASK].keys():
        values = [
            r[scenario][TASK][key] for r in results_list
        ]
        metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    return metrics


def save_aggregated_csv(aggregated, directory="results"):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{TASK}_results_10runs_mean_std.csv")

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'scenario',
            'metric',
            'mean',
            'std'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for scenario, metrics in aggregated.items():
            for metric, values in metrics.items():
                writer.writerow({
                    'scenario': scenario,
                    'metric': metric,
                    'mean': values['mean'],
                    'std': values['std']
                })

    print(f"\nResultados agregados salvos em: {filename}")

if __name__ == "__main__":
    print("\n--- Experimento multi-task 10 execuções ---")

    dataset_source = MultiTaskDataset(
        file_name='datasets/stage/dataset_stage234.csv',
        target_prog='CKD progression',
        target_proteinuria='proteinuria'
    )

    dataset_target = MultiTaskDataset(
        file_name='datasets/stage/dataset_stage5.csv',
        target_prog='CKD progression',
        target_proteinuria='proteinuria'
    )

    tester = SingleTaskTransferTester(
        source_dataset=dataset_source,
        target_dataset=dataset_target
    )

    all_results = []

    for run in range(N_RUNS):
        print(f"\n>>> Execução {run + 1}/{N_RUNS}")
        results = tester.run()
        all_results.append(results)

    aggregated_results = {
        'baseline': aggregate_metrics(all_results, 'baseline'),
        'source_dataset': aggregate_metrics(
            [r['transfer_learning'] for r in all_results],
            'source_dataset'
        ),
        'fine_tuned_target': aggregate_metrics(
            [r['transfer_learning'] for r in all_results],
            'fine_tuned_target'
        )
    }

    save_aggregated_csv(aggregated_results)
