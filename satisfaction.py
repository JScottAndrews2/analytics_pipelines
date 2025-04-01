from modules.data_preparator import DataPreparator
from modules.scale_analyzer import ScaleAnalyzer
from modules.model_pipeline import MLModelsPipeline


if __name__ == '__main__':
    features = ['A1a', 'A1b', 'A2a', 'A2b', 'A2c', 'A2d', 'A2e', 'A2f', 'A2g', 'A2h', 'A3a', 'A3b', 'A3c', 'A3d',
                'A3e', 'A3f', 'A3g', 'A3h', 'A3i', 'A3j', 'A3k', 'A3l', 'A3m', 'A4', 'A4ai', 'A4aii', 'A4aiii',
                'A4aiv', 'A4av', 'A4avi', 'A4avii', 'B1a', 'B1b', 'B2a', 'B2b', 'B2c', 'B2d', 'B3a', 'B3b', 'B3c',
                'B3d', 'B4a', 'B4b', 'B4c', 'B4d', 'B5a', 'B5b', 'B5c', 'B5d', 'B6', 'B7a', 'B7b', 'B7c', 'B7d',
                'B7e', 'B8', 'B8ai', 'B8aii', 'B8aiii', 'B8aiv', 'B8av', 'B8avi', 'B8avii', 'B9', 'C1a', 'C1b', 'C1c',
                'C1d', 'C1e', 'C1f', 'C1g', 'C2', 'C3', 'C4a', 'C4b', 'C4c', 'C4d', 'C4e', 'C4f', 'C4g', 'C4h', 'C4i',
                'C4j', 'C4k', 'C4l', 'c4m', 'C4n', 'C4o', 'C4ii', 'C5', 'C6', 'C6ai', 'C6aii', 'C6aiii', 'C6aiv',
                'C6av', 'C6avi', 'C6avii', 'C6aviii', 'C6aix', 'C6ax', 'C6axi', 'C7', 'C8', 'C9', 'D3', 'D8', 'D14']
    dep_var = 'AgencySize'
    data_prep = DataPreparator(data="data/satisfaction/survey (1).csv", features=features, dep_var=dep_var, max_miss=100)
    data_prep.split_data(val_set=False, test_size=.30, random_state=456)

    sa = ScaleAnalyzer(data_preparator=data_prep)
    efa_factor_loadings = sa.efa(rotation='oblimin', estimation='ml', n_factors=5)
    print(efa_factor_loadings)

    # ---- CFA ---- #
    cfa_output = sa.cfa(scale_dict=sa.scale_dict)
    print(cfa_output)

    # ---- Polytomous IRT (Graded unfolding model)---- #
    irt_stats_res, abilities = sa.irt_gum(scale_dict=sa.scale_dict)
    print(irt_stats_res)
    print(abilities)

    # ---- Creation of Scales ---- #
    scales_data = sa.create_scales(scale_dict=sa.scale_dict, dataset=data_prep.data.drop(labels=[dep_var], axis=1))
    # merge in the independent variable
    scales_data = scales_data.merge(data_prep.data[dep_var].to_frame(), left_index=True, right_index=True)
    scales_data_prep = DataPreparator(data=scales_data, features=list(scales_data.columns[0:-1].values), dep_var=dep_var)
    scales_data_prep.split_data(val_set=False, test_size=.30, random_state=456)

    # ---- Ensemble ---- #
    model_specs = {
        'lr': {},
        'knn': {
           'n_neighbors': 5,
            'leaf_size': 20
        },
        'svr': {
            'kernel': 'rbf'
        },
        'rf': {
            'n_estimators': 50,
            'max_depth': 10
        }
    }
    pipeline = MLModelsPipeline(data_preparator=scales_data_prep, models=model_specs.keys())
    pipeline.train_models(model_specs=model_specs)
    evaluation_results = pipeline.evaluate_performance(scorer=['mean_absolute_error', 'mean_squared_error', 'r2_score'],
                                                       ensemble='regressor')
    print(evaluation_results)


