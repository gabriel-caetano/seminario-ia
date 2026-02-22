import pandas as pd
import os

input_path = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\original\dataset.csv"

output_folder = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\age"
os.makedirs(output_folder, exist_ok=True)

columns_to_keep = [
    "age",
    "etiology of CKD",
    "CKD_stage",
    "proteinuria",
    "SBP",
    "Hb",
    "Alb",
    "eGFR",
    "UPCR",
    "CKD progression",
]

df = pd.read_csv(input_path)
df_filtered = df[columns_to_keep]

# Separação por idade e remoção da coluna age
df_idosos = df_filtered[df_filtered["age"] >= 60].drop(columns=["age"])
df_adultos = df_filtered[df_filtered["age"] < 60].drop(columns=["age"])

output_idosos = os.path.join(output_folder, "dataset_idosos.csv")
df_idosos.to_csv(output_idosos, index=False)

output_adultos = os.path.join(output_folder, "dataset_adultos.csv")
df_adultos.to_csv(output_adultos, index=False)

print("Processo concluído com sucesso.")
