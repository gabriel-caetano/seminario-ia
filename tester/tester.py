from dataset import Dataset
import time
import sys
from dataset import Dataset
from mlp import MLP

class Tester:
    def __init__(self, target_column, source_dataset, target_dataset=None, name="MLP Tester"):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.target_column = target_column
        self.model_history = []
        self.name = name

        if not isinstance(self.source_dataset, Dataset):
            raise TypeError("source_dataset must be an instance of the Dataset class")
        if target_dataset is not None and not isinstance(self.target_dataset, Dataset):
            raise TypeError("target_dataset must be an instance of the Dataset class")

    def run(self):
        start_time = time.time()
        print(f"\n--- Running Tester: {self.name} ---")
        model = MLP(self.source_dataset.get_shape())
        # print("\n--- Model Summary ---")
        # model.summary()
        # first train
        model.train(self.source_dataset, name="Source Dataset Training" + " " + self.name)
    
        if self.target_dataset is not None:
            # evaluate on target dataset before transfer learning
            # print("\n--- Evaluating on Target Dataset before Transfer Learning ---")
            source_stats_target = model.predict(self.target_dataset)
            print("\n--- Transfer Learning on Target Dataset ---")
            # Freeze layers except the last one
            for layer in model.model.layers[:3]:
                if 'dropout' not in layer.name:
                    # print(f"Freezing layer: {layer.name}")
                    layer.trainable = False
                

            # Recompile the model to apply the changes
            model.compile(learning_rate=0.0001)

            model.train(self.target_dataset, name="Transfer Learning Training")

            tl_stats_target = model.predict(self.target_dataset)
            self.model_history.append(model)
            
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"\n--- Execution time: {execution_time:.2f} seconds ---")
            
            return {
                'source_stats': source_stats_target,
                'tl_stats': tl_stats_target
            }
        else:
            source_stats_target = model.predict(self.source_dataset)
            self.model_history.append(model)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"\n--- Execution time: {execution_time:.2f} seconds ---")
            return {
                'target_stats': source_stats_target
            }
        

if __name__ == "__main__":
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
