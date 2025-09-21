import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from collections import namedtuple
scaler = StandardScaler()

RANDOM_STATE=42
TEST_SIZE=0.2

class Dataset:
    def __init__(self, file_name, target_column=None):
        # load file to dataset
        self.ds = pd.read_csv(file_name)
        self.target_column = target_column  

        if target_column is not None:
            # split features(x) and target(y)
            self.features = self.ds.drop(target_column, axis=1)
            self.target = self.ds[target_column]
            # split into train and test sets
            # by default with test_size=0.2
            self.split()

    def setTargetColumn(self, target_column):
        self.target_column = target_column  
        # split features(x) and target(y)
        self.features = self.ds.drop(target_column, axis=1)
        self.target = self.ds[target_column]
        # split into train and test sets
        # by default with test_size=0.2
        self.split()

    def __repr__(self):
        return (f"Dataset(shape={self.get_shape()}")

    def split(self, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        #  splitting
        features_train, features_test, self.target_train, self.target_test = train_test_split(
            self.features,
            self.target,
            test_size=test_size,
            random_state=random_state
        )
        # scaling
        self.features_train = scaler.fit_transform(features_train)
        self.features_test = scaler.transform(features_test)

    def get_shape(self):
        return self.features.shape[1]

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Error: Please provide the dataset path as the first argument.")
        sys.exit(1)
    file_name = sys.argv[1]
    if not isinstance(file_name, str) or not file_name.endswith('.csv'):
        print("Error: The dataset path must be a string ending with '.csv'.")
        sys.exit(1)
    
    dataset = Dataset(file_name, 'CKD progression')
    print(dataset)
    print(dataset.get_shape())