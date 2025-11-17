from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
scaler = StandardScaler()

def scale(csv_path):
    file_name = csv_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(csv_path.split('/')[:-1])
    file_extension = csv_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")
    df = pd.read_csv(csv_path)
    target_col = "CKD progression"
    features = df.drop(columns=[target_col])
    scaled_features = scaler.fit_transform(features)
    scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_data[target_col] = df[target_col]
    scaled_data = scaled_data.reindex(columns=df.columns)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    base, ext = os.path.splitext(csv_path)
    new_path = f"{file_location}/scale/{file_name}_scaled.{ext}"
    scaled_df.to_csv(new_path, index=False)
