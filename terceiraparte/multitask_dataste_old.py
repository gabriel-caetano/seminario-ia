import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
RANDOM_STATE = 42


class MultiTaskDataset:
    def __init__(self, file_name, target_prog="CKD progression",
                 train_size=0.6, val_size=0.2, test_size=0.2):
        """
        Classe para carregar e dividir datasets apenas para CKD progression (single-task)
        """
        self.ds = pd.read_csv(file_name)

        self.target_prog = target_prog

        for col in self.ds.columns:
            if col == target_prog:
                continue

            unique_values = self.ds[col].nunique()
            if unique_values <= 10:
                continue
            else:
                self.ds[col] = self.ds[col].astype(float)
                self.ds[col] = scaler.fit_transform(
                    self.ds[col].values.reshape(-1, 1)
                ).flatten()

        self.features = self.ds.drop([target_prog], axis=1)
        self.target = self.ds[target_prog]

        self.split(train_size, val_size, test_size)

    def split(self, train_size, val_size, test_size):
        X_train_val, self.features_test, y_train_val, self.target_test = train_test_split(
            self.features,
            self.target,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=self.target,
        )

        val_size_adjusted = val_size / (train_size + val_size)

        self.features_train, self.features_validation, self.target_train, self.target_validation = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size_adjusted,
            random_state=RANDOM_STATE,
            stratify=y_train_val,
        )

    def get_shape(self):
        return self.features.shape[1]

    def __repr__(self):
        return f"MultiTaskDataset(shape={self.get_shape()})"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Erro: forneça o caminho do dataset CSV.")
        sys.exit(1)

    dataset = MultiTaskDataset(sys.argv[1])
    print(dataset)
    print("N features:", dataset.get_shape())
