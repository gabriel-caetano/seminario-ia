import pandas as pd
import os

input_path = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\original\dataset.csv"

output_folder = r"C:\Users\rafae\seminario-ia\terceiraparte\datasets\stage"
os.makedirs(output_folder, exist_ok=True)

columns_to_keep = [
    "age",
    "SBP",
    "etiology of CKD",
    "Hb",
    "Alb",
    "eGFR",
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

df_stage234 = df_filtered[df_filtered['CKD_stage'].isin([2, 3, 4])]
output_stage234 = os.path.join(output_folder, "dataset_stage234.csv")
df_stage234.to_csv(output_stage234, index=False)

df_stage5 = df_filtered[df_filtered['CKD_stage'] == 5]
output_stage5 = os.path.join(output_folder, "dataset_stage5.csv")
df_stage5.to_csv(output_stage5, index=False)

print(f"Datasets separados por CKD_stage salvos em:\n{output_stage234}\n{output_stage5}")
