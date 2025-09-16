import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from collections import namedtuple
scaler = StandardScaler()

RANDOM_STATE=42
TEST_SIZE=0.2

class Dataset:

    def __init__(self, file_name):
        # load file to dataset
        self.ds = pd.read_csv(file_name)

        # split features(x) and target(y)
        self.features = self.ds.drop('CKD progression', axis=1)
        self.target = self.ds['CKD progression']
        
        # split into train and test sets
        # by default with test_size=0.2
        self.split()

    def __repr__(self):
        return (f"Dataset(X_train_shape={self.X_train.shape}, "
            f"X_test_shape={self.X_test.shape}, "
            f"y_train_shape={self.y_train.shape}, "
            f"y_test_shape={self.y_test.shape})")

    def split(self, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        #  splitting
        features_train, features_test, self.target_train, self.target_test = train_test_split(
            self.features, self.target, test_size=test_size, random_state=random_state
        )
        # scaling
        self.features_train = scaler.fit_transform(features_train)
        self.features_test = scaler.transform(features_test)

    def getShape(self):
        return self.features.shape[1]

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Error: Please provide the dataset path as the first argument.")
        sys.exit(1)
    file_name = sys.argv[1]
    if not isinstance(file_name, str) or not file_name.endswith('.csv'):
        print("Error: The dataset path must be a string ending with '.csv'.")
        sys.exit(1)
    
    dataset = Dataset(file_name)
    print(dataset)
    print(dataset.X_train.shape)
    print(dataset.X_test.shape)
    print(dataset.y_train.shape)
    print(dataset.y_test.shape)