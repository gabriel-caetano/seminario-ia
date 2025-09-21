from boruta_test import apply_boruta
from fill_missing_values import fill_missing_values
from split_by_age import split_by_age
from split_by_etiology import split_by_etiology
from split_by_stage import split_by_stage

if __name__ == "__main__":
    fill_missing_values('datasets/dataset.csv', 'CKD progression')
    apply_boruta('datasets/dataset_filled.csv', 'CKD progression')
    split_by_age(dataset_path='datasets/dataset_filled_boruta.csv')
    split_by_etiology(dataset_path='datasets/dataset_filled_boruta.csv')
    split_by_stage(dataset_path='datasets/dataset_filled_boruta.csv')