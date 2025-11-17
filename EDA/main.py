# from boruta_test import apply_boruta
from fill_missing_values import fill_missing_values
from split_by_age import split_by_age
from split_by_etiology import split_by_etiology
from split_by_stage import split_by_stage
from plot_jsd import plot_jsd
from plot_distribution import plot_distribution
from scale import scale

if __name__ == "__main__":
    # fill_missing_values('datasets/dataset.csv', 'CKD progression')
    
    # apply_boruta('datasets/filled/dataset_filled.csv', 'CKD progression')
    
    # split and drop column used for splitting
    # split_by_age(dataset_path='datasets/filled/boruta/dataset_filled_boruta.csv')
    # split_by_etiology(dataset_path='datasets/filled/boruta/dataset_filled_boruta.csv')
    # split_by_stage(dataset_path='datasets/filled/boruta/dataset_filled_boruta.csv')

    # plot distributions of the resulting datasets
    plot_distribution(
        'datasets/filled/boruta/age/dataset_filled_boruta_age>=60.csv'
        ,'datasets/filled/boruta/age/dataset_filled_boruta_age<60.csv'
    )
    # plot_jsd(
    #     'datasets/filled/boruta/age/dataset_filled_boruta_age>=60.csv',
    #     'datasets/filled/boruta/age/dataset_filled_boruta_age<60.csv'
    # )

    plot_distribution(
        'datasets/filled/boruta/etiology/dataset_filled_boruta_etiology12.csv'
        ,'datasets/filled/boruta/etiology/dataset_filled_boruta_etiology34.csv'
    )
    # plot_jsd(
    #     'datasets/filled/boruta/etiology/dataset_filled_boruta_etiology12.csv',
    #     'datasets/filled/boruta/etiology/dataset_filled_boruta_etiology34.csv'
    # )

    plot_distribution(
        'datasets/filled/boruta/stage/dataset_filled_boruta_stage234.csv'
        ,'datasets/filled/boruta/stage/dataset_filled_boruta_stage5.csv'
    )
    # plot_jsd(
    #     'datasets/filled/boruta/stage/dataset_filled_boruta_stage5.csv',
    #     'datasets/filled/boruta/stage/dataset_filled_boruta_stage234.csv'
    # )

    # scale all remaining features for training
    
    # scale('datasets/filled/boruta/age/dataset_filled_boruta_age>=60.csv')
    # scale('datasets/filled/boruta/age/dataset_filled_boruta_age<60.csv')
    # scale('datasets/filled/boruta/etiology/dataset_filled_boruta_etiology12.csv')
    # scale('datasets/filled/boruta/etiology/dataset_filled_boruta_etiology34.csv')
    # scale('datasets/filled/boruta/stage/dataset_filled_boruta_stage5.csv')
    # scale('datasets/filled/boruta/stage/dataset_filled_boruta_stage234.csv')
