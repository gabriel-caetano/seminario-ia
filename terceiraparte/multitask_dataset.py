import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class MultiTaskDataset:
    def __init__(
        self,
        folder_name,
        dataset_name,
        target_prog="CKD progression",
        target_proteinuria="proteinuria",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    ):
        self.train_ds = pd.read_csv(f"{folder_name}/{dataset_name}/train.csv")
        self.val_ds = pd.read_csv(f"{folder_name}/{dataset_name}/val.csv")
        self.test_ds = pd.read_csv(f"{folder_name}/{dataset_name}/test.csv")
        
        self.target_prog = target_prog
        self.target_proteinuria = target_proteinuria

        self.features_train = self.train_ds.drop([target_prog, target_proteinuria], axis=1)
        self.target_train = self.train_ds[target_prog]
        self.proteinuria_train = self.train_ds[target_proteinuria]

        self.features_validation = self.val_ds.drop([target_prog, target_proteinuria], axis=1)
        self.target_validation = self.val_ds[target_prog]
        self.proteinuria_validation = self.val_ds[target_proteinuria]

        self.features_test = self.test_ds.drop([target_prog, target_proteinuria], axis=1)
        self.target_test = self.test_ds[target_prog]
        self.proteinuria_test = self.test_ds[target_proteinuria]

    def get_shape(self):
        return self.features_train.shape[1]

    def __repr__(self):
        return (
            f"MultiTaskDataset(shape={self.get_shape()}, "
            f"tasks=['prog', 'proteinuria'])"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit(1)

    dataset = MultiTaskDataset(sys.argv[1], sys.argv[2])
