import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from dataset import Dataset
from tester import Tester
import sys

TEST_SIZE=0.2
RANDOM_STATE=42


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Error: Please provide the dataset path as the first argument.")
    #     sys.exit(1)
    # file_name = sys.argv[1]
    # if not isinstance(file_name, str) or not file_name.endswith('.csv'):
    #     print("Error: The dataset path must be a string ending with '.csv'.")
    #     sys.exit(1)
    
    # full dataset
    idosos = Dataset('datasets/dataset_filled_boruta_age>=60.csv', 'CKD progression')
    adultos = Dataset('datasets/dataset_filled_boruta_age<60.csv', 'CKD progression')
    
    bl_1 = Tester('CKD progression', source_dataset=adultos)
    tl_1 = Tester('CKD progression', source_dataset=idosos, target_dataset=adultos)

    print("\n\n--- Resultados experimento 1 ---")
    print("\n###########################################")
    print("Baseline:")
    print("\n###########################################")
    res1 = bl_1.run()
    print(res1)
    print("\n###########################################")
    print("\nTransfer Learning:")
    print("\n###########################################")
    res2 = tl_1.run()
    print(res2)

    


# bl
# Acurácia : 80.00%
# Precisão : 55.56%
# Recall   : 83.33%
# F1-score : 66.67%
# tl
# Acurácia : 84.00%
# Precisão : 62.50%
# Recall   : 83.33%
# F1-score : 71.43%