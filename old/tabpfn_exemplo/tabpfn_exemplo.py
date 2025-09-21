import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from tabpfn import TabPFNClassifier

# Carregar o dataset
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Separar features e target
X = df.drop('death', axis=1)
y = df['death']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar o cliente TabPFN
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))

# Calcular e exibir o log loss
loss = log_loss(y_test, prediction_probabilities)
print("Log Loss:", loss)
