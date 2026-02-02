import pandas as pd
import os

input_path = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\original\dataset.csv"

output_folder = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\age"
os.makedirs(output_folder, exist_ok=True)

columns_to_keep = [
    "age",
    "SBP",
    "etiology of CKD",
    "Hb",
    "Alb",
    "CKD_stage",
    "proteinuria",
    "UPCR",
    "CKD progression",
    "RRT"
]

df = pd.read_csv(input_path)

df_filtered = df[columns_to_keep]

output_full = os.path.join(output_folder, "dataset_preprocessed.csv")
df_filtered.to_csv(output_full, index=False)
print(f"Dataset pré-processado completo salvo em: {output_full}")

df_idosos = df_filtered[df_filtered["age"] >= 60]
output_idosos = os.path.join(output_folder, "dataset_idosos.csv")
df_idosos.to_csv(output_idosos, index=False)

df_adultos = df_filtered[df_filtered["age"] < 60]
output_adultos = os.path.join(output_folder, "dataset_adultos.csv")
df_adultos.to_csv(output_adultos, index=False)

print(
    f"Datasets separados por idade salvos em:\n"
    f"{output_idosos}\n"
    f"{output_adultos}"
)
