from dataset import Dataset

# import sys
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from dataset import Dataset
from mlp import MLP

class Tester:
    def __init__(self, target_column, source_dataset, target_dataset=None):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.target_column = target_column
        self.model_history = []

        if not isinstance(self.source_dataset, Dataset):
            raise TypeError("source_dataset must be an instance of the Dataset class")
        if target_dataset is not None and not isinstance(self.target_dataset, Dataset):
            raise TypeError("target_dataset must be an instance of the Dataset class")

    def run(self):
        model = MLP(self.source_dataset.get_shape())

        # first train
        model.train(self.source_dataset)

        # evaluate on source dataset
        model.predict(self.source_dataset)
        self.model_history.append(model)
        if self.target_dataset is not None:
            # evaluate on target dataset before transfer learning
            model.predict(self.target_dataset)
            print("\n--- Transfer Learning on Target Dataset ---")
            # Freeze layers except the last one
            for layer in model.model.layers[:-2]:
                layer.trainable = False

            # Recompile the model to apply the changes
            model.compile(learning_rate=0.01)  # You might want to use a lower learning rate for fine-tuning

            model.train(self.target_dataset)

            model.predict(self.target_dataset)
            self.model_history.append(model)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Error: Please provide at least one dataset path as argument.")
        sys.exit(1)
    
    dataset_paths = sys.argv[1:]
    if len(dataset_paths) > 2:
        print("Error: Please provide at most two dataset paths.")
        sys.exit(1)
    dataset_source = Dataset(sys.argv[1], target_column='CKD progression')
    dataset_target = Dataset(sys.argv[2], target_column='CKD progression') if len(dataset_paths) == 2 else None

    tester = Tester('CKD progression', source_dataset=dataset_source, target_dataset=dataset_target)
    tester.run()