import os
import sys
import json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dataset
from tester import Tester
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
    source_dataset = Dataset('ds/dataset_filled_boruta_age>=60_scaled.csv', 'CKD progression')
    target_dataset = Dataset('ds/dataset_filled_boruta_age<60_scaled.csv', 'CKD progression')

    target_test = Tester('CKD progression', source_dataset=target_dataset, name="Age Target Tester")
    tl_test = Tester('CKD progression', source_dataset=source_dataset, target_dataset=target_dataset, name="Age Transfer Learning Tester")

    print("\n\n--- Experimento com divisão por idade ---")
    target_res = target_test.run()
    tl_res = tl_test.run()

    with open('./results/result.json', 'w') as fp:
        json.dump({
            'source': tl_res.get('source_stats'),
            'target': target_res.get('target_stats'),
            'transfer_learning': tl_res.get('tl_stats'),
        }, fp)



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