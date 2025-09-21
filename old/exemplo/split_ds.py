import pandas as pd

# Define filenames
input_file = 'dataset.csv'
source_file = 'dataset_source.csv'
target_file = 'dataset_target.csv'

# Load the main dataset
df = pd.read_csv(input_file)
print(f"Successfully loaded '{input_file}' with {len(df)} records.")

# --- Splitting Logic ---
# We will split the dataset into two groups based on the median age.
# This simulates two different populations, ideal for a transfer learning scenario.
# One group (source) will be the "younger" population, and the other (target) the "older" one.
split_column = 'age' # Assuming 'age4' was a typo for 'age'

split_value = 60
print(f"Splitting the dataset by the median age: {split_value:.2f} years.")

# Create the source dataset (age <= median)
source_df = df[df[split_column] >= split_value].copy()

# Create the target dataset (age > median)
target_df = df[df[split_column] < split_value].copy()

# Save the new datasets to CSV files
source_df.to_csv(source_file, index=False)
target_df.to_csv(target_file, index=False)

print(f"\nSuccessfully created the following files:")
print(f"- '{source_file}' with {len(source_df)} records (age <= {split_value:.2f}).")
print(f"- '{target_file}' with {len(target_df)} records (age > {split_value:.2f}).")
