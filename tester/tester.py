import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# receive one or 2 datasets as argument
# if 1 dataset, train on it and test on it
# if 2 datasets, train on the first and test on the second
# at the end show the following metrics of the test:
# accuracy
# precision
# recall
# f1-score
# confusion matrix
# classification report
from dataset import load_dataset  # Assumes load_dataset returns (X, y)
from mlp import create_mlp_model  # Assumes create_mlp_model returns a compiled Keras model

def main():
    args = sys.argv[1:]
    if len(args) == 1:
        train_path = test_path = args[0]
    elif len(args) == 2:
        train_path, test_path = args
    else:
        print("Usage: python tester.py <train_dataset> [<test_dataset>]")
        sys.exit(1)

    # Load datasets
    X_train, y_train = load_dataset(train_path)
    X_test, y_test = load_dataset(test_path)

    # Create and train model
    model = create_mlp_model(input_shape=X_train.shape[1], num_classes=len(set(y_train)))
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1) if y_pred.ndim > 1 else (y_pred > 0.5).astype(int)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred_classes))
    print("Precision:", precision_score(y_test, y_pred_classes, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred_classes, average='weighted'))
    print("F1-score:", f1_score(y_test, y_pred_classes, average='weighted'))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))
    print("Classification Report:\n", classification_report(y_test, y_pred_classes))

    # Save the model
    model.save("model.h5")

if __name__ == "__main__":
    main()
