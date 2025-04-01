import inspect
import pandas as pd
from sklearn import metrics
from typing import Dict, Union
from modules.model_classes import Model
from modules.data_preparator import DataPreparator
from sklearn.model_selection import GridSearchCV


class MLModelsPipeline:
    """
    A class that creates the pipeline for model training, cross validation, and testing
    """
    def __init__(self, data_preparator: DataPreparator, models):
        self.model_names = models
        self.data_preparator = data_preparator
        self.models = self._get_model_classes()
        self.trained_models = None
        self.model_specs = None
        self.models_train_pred = None

    def cross_validate_models(self, scorer, models_search_params: Dict, **kwargs):
        """
        A method to run grid search cross validation to identify the best model hyper-parameters

        :param scorer: str, callable, list, tuple, dict, None
            The evaluation metric(s) for model performance

        :param models_search_params: Dict
            A dictionary of model names and grid search parameter values for hyper-parameter
            tuning

        :param kwargs:
            keyword arguments for GridSearchCV

        :return: Dict
            Results from grid search
        """
        self.cv_results = {}
        self.best_params = {}
        for model_name, model_class in self.models.items():
            # model_kwargs = self.model_specs[model_name]
            model = model_class(data_preparator=self.data_preparator)
            cv = GridSearchCV(estimator=model, param_grid=models_search_params[model_name], scoring=scorer, verbose=1,
                              **kwargs)
            cv.fit(self.data_preparator.x_train, self.data_preparator.y_train)
            self.cv_results[model_name] = cv.cv_results_
            self.best_params[model_name] = cv.best_params_
            print(f"Cross-validation for {model_name} has completed!")

    def train_models(self, model_specs: Dict):
        """"
        :param model_specs: Dict
            A dictionary containing model names as keys and a Dict of hyper-parameter settings as items
        """
        self.model_specs = model_specs
        self.trained_models = {}
        for model_name, model_class in self.models.items():
            model_kwargs = self.model_specs[model_name]
            model = model_class(data_preparator=self.data_preparator, **model_kwargs)
            model.fit()
            self.trained_models[model_name] = model
            print(f"Training for model {model_name} is complete!")

    def evaluate_performance(self, scorer, ensemble: Union[str, None]= None):
        """
        A method for evaluating models performance

        :param scorer: str, list, callable
            A method for scoring model predictions. If a str or list is provided, it must correspond to a class from
            sklearn.metrics

        :param ensemble: str or None
            A string for the modeling type: classifier or regressor. If none, no ensemble is created.

        :return: pd.DataFrame
            A dataframe containing model performance evaluations on training and testing dataset for each model
        """
        if self.trained_models is None:
            raise ValueError("There are no trained models stored in this class. Call the train_models method first.")
        self.models_train_pred = {}
        self.models_test_pred = {}
        results = {}
        for model_name, trained_model in self.trained_models.items():
            train_pred = trained_model.predict(X=self.data_preparator.x_train)
            test_pred = trained_model.predict(X=self.data_preparator.x_test)
            # test_predictions.append(model.predictions)
            train_perf, test_perf = self._score_model(scorer=scorer, train_pred=train_pred, test_pred=test_pred)
            results[model_name] = {'train': train_perf, 'test': test_perf}

            self.models_train_pred[model_name] = train_pred
            self.models_test_pred[model_name] = test_pred
            print(f"Evaluation for model {model_name} is complete!")
        if isinstance(ensemble, str):
            train_predictions = pd.DataFrame(self.models_train_pred)
            test_predictions = pd.DataFrame(self.models_test_pred)
            if ensemble.lower() == 'regressor':
                ens_train_pred = train_predictions.mean(axis=1)
                ens_test_pred = test_predictions.mean(axis=1)

            elif ensemble.lower() == 'classifier':
                ens_train_pred = train_predictions.mode(axis=1)
                ens_test_pred = test_predictions.mode(axis=1)
            else:
                raise ValueError(f"ensemble must be one of regressor, classifier, or None, but {ensemble} was provided")
            self.models_train_pred['ensemble'] = ens_train_pred.values.reshape(-1)
            self.models_test_pred['ensemble'] = ens_test_pred.values.reshape(-1)
            train_perf, test_perf = self._score_model(scorer=scorer, train_pred=ens_train_pred, test_pred=ens_test_pred)
            results['ensemble'] = {'train': train_perf, 'test': test_perf}

        out = pd.DataFrame.from_dict(results, orient="index").stack().to_frame()
        out = pd.DataFrame(out[0].values.tolist(), index=out.index)
        return out

    def evaluate_fairness(self, scorer, comparison_dict: Dict):
        """

        :param scorer: str
            A scoring method. Currently, only the string 'disparate_impact' is accepted.

        :param comparison_dict: Dict
            A dicationary containing the demographic variables as initial keys, followed by a dictionary with majority
            group as a key and a list of other groups for comaprison. The majority group (denominator) will be listed
            first in the output table

        :return: pd.DataFrame
            a dataframe containing fairness scores for each model and each pairwise groups comparisons
        """
        if self.trained_models is None:
            raise ValueError("There are no trained models stored in this class. Call the train_models method first.")
        if self.models_train_pred is None:
            raise ValueError("Model performance has not been assessed. Call the evaluate_performance method first.")

        demo_train_data = pd.concat([pd.DataFrame(self.models_train_pred), pd.DataFrame(self.data_preparator.y_train), self.data_preparator.d_train],
                                    axis=1)
        demo_test_data = pd.concat([pd.DataFrame(self.models_test_pred), pd.DataFrame(self.data_preparator.y_test), self.data_preparator.d_test],
                                   axis=1)
        results = {}
        for model_name, _ in self.trained_models.items():
            results[model_name] = {}
            for demo_var in self.data_preparator.demo_vars:
                if scorer == 'disparate_impact':
                    grps = demo_train_data[[demo_var] + [model_name]].groupby([demo_var] + [model_name]).size().unstack(fill_value=0).stack().reset_index(name='count')
                for grp1_name, values in comparison_dict[demo_var].items():
                    grp1 = grps[grps[demo_var]==grp1_name]
                    grp1_total = grp1['count'].sum(axis=0)
                    grp1_pos = grp1[grp1[model_name]==1]['count'].values[0]
                    grp1_ratio = grp1_pos/grp1_total
                    for grp2_name in values:
                        grp2 = grps[grps[demo_var] == grp2_name]
                        grp2_total = grp2['count'].sum(axis=0)
                        grp2_pos = grp2[grp2[model_name] == 1]['count'].values[0]
                        grp2_ratio = grp2_pos / grp2_total
                        results[model_name][f"{grp1_name}_{grp2_name}"] = grp2_ratio / grp1_ratio
        results = pd.DataFrame(results).T
        return results

    def _get_model_classes(self):
        """
        An internal method for creating the list of model objects

        :return: List
            A list contianing model objects
        """
        available_models = [sub_model for model in Model.__subclasses__() for sub_model in model.__subclasses__()]
        models = {}
        for model_name in self.model_names:
            # Dynamic model selection using abstract factory
            if model_name.lower() in [model_type.__name__.lower() for model_type in available_models]:
                model_class = [model_type for model_type in available_models if model_type.__name__.lower() == model_name.lower()]
            else:
                raise ValueError(f"model name [{model_name}] is not available.")
            models[model_name] = model_class[0]
        return models

    def _score_model(self, train_pred, test_pred, scorer):
        """
        An intenral method for coring a model based on prediction sets.

        :param train_pred: np.array
            An np.array of model prediction for the training dataset

        :param test_pred: np.array
            An np.array of model prediction for the test dataset

        :param scorer: str, list, callable
            A method for scoring model predictions. If a str or list is provided, it must correspond to a class from
            sklearn.metrics

        :return: dict, dict
        """
        if isinstance(scorer, str):
            metric = [metric for name, metric in inspect.getmembers(metrics) if name.lower() == scorer.lower()]
            train_performance = metric[0](self.data_preparator.y_train, train_pred)
            test_performance = metric[0](self.data_preparator.y_test, test_pred)
            return train_performance, test_performance

        elif isinstance(scorer, list):
            train_performance = {}
            test_performance = {}
            for metric_name in scorer:
                metric = [metric for name, metric in inspect.getmembers(metrics) if name.lower() == metric_name.lower()]
                if len(metric) == 0:
                    raise ValueError(f"The selected metric name [{metric_name}] was not found. You must provide the "
                                     "name of a metrics from the sklearn.metrics classes")
                train_performance[metric_name] = metric[0](self.data_preparator.y_train, train_pred)
                test_performance[metric_name] = metric[0](self.data_preparator.y_test, test_pred)
            return train_performance, test_performance

        elif callable(scorer):
            train_performance = scorer(self.data_preparator.y_train, train_pred)
            test_performance = scorer(self.data_preparator.y_test, test_pred)
            return train_performance, test_performance
