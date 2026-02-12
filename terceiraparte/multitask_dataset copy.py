import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class MultiTaskDataset:
    def __init__(
        self,
        file_name,
        target_prog="CKD progression",
        target_proteinuria="proteinuria",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
    ):
        self.ds = pd.read_csv(file_name)
        
        self.target_prog = target_prog
        self.target_proteinuria = target_proteinuria

        self.features = self.ds.drop([target_prog, target_proteinuria], axis=1)
        self.target = self.ds[target_prog]
        self.proteinuria = self.ds[target_proteinuria]

        self.split(train_size, val_size, test_size)
        self.scale_features()

    def split(self, train_size, val_size, test_size):
        (
            X_train_val,
            self.features_test,
            y_train_val,
            self.target_test,
            prot_train_val,
            self.proteinuria_test,
        ) = train_test_split(
            self.features,
            self.target,
            self.proteinuria,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=self.target,
        )

        val_size_adjusted = val_size / (train_size + val_size)

        (
            self.features_train,
            self.features_validation,
            self.target_train,
            self.target_validation,
            self.proteinuria_train,
            self.proteinuria_validation,
        ) = train_test_split(
            X_train_val,
            y_train_val,
            prot_train_val,
            test_size=val_size_adjusted,
            random_state=RANDOM_STATE,
            stratify=y_train_val,
        )

    def scale_features(self):
        numeric_cols = []

        for col in self.features_train.columns:
            nunique = self.features_train[col].nunique(dropna=True)
            if nunique > 10:
                numeric_cols.append(col)

        if numeric_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(self.features_train[numeric_cols].astype(float))

            self.features_train.loc[:, numeric_cols] = self.scaler.transform(
                self.features_train[numeric_cols].astype(float)
            )
            self.features_validation.loc[:, numeric_cols] = self.scaler.transform(
                self.features_validation[numeric_cols].astype(float)
            )
            self.features_test.loc[:, numeric_cols] = self.scaler.transform(
                self.features_test[numeric_cols].astype(float)
            )

    def get_shape(self):
        return self.features.shape[1]

    def __repr__(self):
        return (
            f"MultiTaskDataset(shape={self.get_shape()}, "
            f"tasks=['prog', 'proteinuria'])"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit(1)

    dataset = MultiTaskDataset(sys.argv[1])
