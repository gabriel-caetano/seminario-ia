import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score

# =========================
# Dataset
# =========================
dataset_path = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\stage\dataset_preprocessed.csv"
df = pd.read_csv(dataset_path)

# =========================
# Features de input
# =========================
features = ["age", "SBP", "etiology of CKD", "Hb", "Alb", "eGFR", "CKD_stage", "proteinuria", "UPCR", "RRT"]

# Task principal
primary_target = "CKD progression"

# Candidate tasks secundárias: todas as features de input
candidate_secondary_targets = features.copy()

print(f"Candidatas a task secundária: {candidate_secondary_targets}")

# =========================
# Avaliar cada candidate secondary task
# =========================
results = []

for target in candidate_secondary_targets:
    # Inputs = todas as features exceto a candidata
    X = df[[f for f in features if f != target]]
    y = df[target]

    # Separar treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Determinar se é contínuo ou discreto
    if y.nunique() > 5:  # arbitrário: >5 valores → regressão
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        results.append({
            "secondary_task": target,
            "type": "regression",
            "predictive_score": score
        })
    else:  # classificação
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if len(y.unique()) == 2:
            y_prob = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        results.append({
            "secondary_task": target,
            "type": "classification",
            "predictive_score": f1,  # usamos F1 como métrica principal
            "accuracy": acc,
            "auc_roc": auc
        })

# =========================
# Ranking das melhores tasks secundárias
# =========================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="predictive_score", ascending=False)

print("\n=== Ranking das possíveis tasks secundárias para TL multi-task ===")
print(results_df)

# =========================
# Salvar CSV
# =========================
results_df.to_csv(r"C:\Users\rafae\seminario-ia\terceiraparte\results\secondary_task_ranking.csv", index=False)
print("\nRanking salvo em: results/secondary_task_ranking.csv")
