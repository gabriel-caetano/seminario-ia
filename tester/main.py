import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dataset import Dataset
from mlp import MLP
import sys

TEST_SIZE=0.2
RANDOM_STATE=42


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide the dataset path as the first argument.")
        sys.exit(1)
    file_name = sys.argv[1]
    if not isinstance(file_name, str) or not file_name.endswith('.csv'):
        print("Error: The dataset path must be a string ending with '.csv'.")
        sys.exit(1)
    
    dataset = Dataset(file_name)
    # print(dataset)
    # print(dataset.X_train.shape)
    # print(dataset.X_test.shape)
    # print(dataset.y_train.shape)
    # print(dataset.y_test.shape)

    model = MLP(
        shape=dataset.getShape(),
        layers=[4,2],
        activation='tanh'
    )
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        learning_rate=0.01
    )

    model.train(dataset)


