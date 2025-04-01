import pandas as pd
import numpy as np
import os
from modules.data_preparator import DataPreparator
from modules.model_pipeline import MLModelsPipeline


if __name__ == '__main__':
    # data = pd.read_csv(os.path.join("data", "census", "adult.csv"))
    # # replace the question mark values with missing
    # data.replace(to_replace='?', value=np.nan, inplace=True)
    # data.replace({'income': {'<=50K': 0, '>50K': 1}}, inplace=True)
    # ---- Dataset paramaters ---- #
    features = ['workclass', 'fnlwgt', 'education', 'educational-num',
                'marital-status', 'occupation', 'relationship',
                'capital-gain', 'capital-loss', 'hours-per-week']
    dep_var = 'income'
    demo_vars = ['race', 'gender']

    # ---- We use this first run to split the data for us, then we save those outputs so that we have a fully
    # ---- reproducible datasets. We can comment out this code, because now we will on the pre-saved datasets with
    # ---- missing data already removed
    # data_prep = DataPreparator(data=data, features=features, dep_var=dep_var, demo_vars=demo_vars, max_miss=None)
    # data_prep.split_data(val_set=False, test_size=0.30, random_state=456)
    # data_prep.encode_categorical(strategy='TargetEncoder')
    # data_prep.x_train.columns
    # data_prep.features
    # data_prep.impute_missing(strategy='knn', n_neighbors=15)
    # data_prep.data.to_csv(os.path.join("data", "census", "data_pre_processed.csv"))
    # pd.concat([data_prep.x_train, data_prep.y_train, data_prep.d_train], axis=1).to_csv(os.path.join("data", "census",
    #                                                                                                  "train.csv"))
    # pd.concat([data_prep.x_test, data_prep.y_test, data_prep.d_test], axis=1).to_csv(os.path.join("data", "census",
    #                                                                                               "test.csv"))

    data_path = os.path.join("data", "census", "data_pre_processed.csv")
    train_path = os.path.join("data", "census", "train.csv")
    test_path = os.path.join("data", "census", "test.csv")
    data_prep = DataPreparator(data=data_path, train_data=train_path, test_data=test_path, features=features,
                               dep_var=dep_var, demo_vars=demo_vars)
    model_grid_params = {
        'LGR': {'penalty': ['l1', 'l2'], 'C': [0.5, 1.0, 2.0], 'solver': ['liblinear']},
        'SVC': {'C': [0.5, 1.0], 'kernel': ['linear', 'poly']},
        'KNNC': {'n_neighbors': [5, 50], 'leaf_size': [10, 30], 'p': [1, 2]},
        'RFC': {'n_estimators': [100, 200], 'max_depth': [50, 100]},
        'MLPC': {'hidden_layer_sizes': [(100,), (20, 50, 20)], 'activation': ['relu', 'logistic']}
    }

    model_evaluation_metrics = ['accuracy', 'precision', 'recall']
    pipeline = MLModelsPipeline(data_preparator=data_prep, models=model_grid_params.keys())
    pipeline.cross_validate_models(scorer='accuracy', models_search_params=model_grid_params,
                                   cv=2, return_train_score=True, refit='accuracy')
    pd.DataFrame(pipeline.cv_results['LGR'])
    print(pipeline.best_params)
    pipeline.train_models(model_specs=pipeline.best_params)
    perf_results = pipeline.evaluate_performance(scorer=['accuracy_score', 'recall_score'], ensemble='classifier')
    print(perf_results)
    comparison_dict = {
        'race': {'White': ['Black', 'Other']},
        'gender': {'Male': ['Female']}
    }
    fairness_results = pipeline.evaluate_fairness(scorer='disparate_impact', comparison_dict=comparison_dict)
    print(fairness_results)
