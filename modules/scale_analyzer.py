import pandas as pd
import numpy as np
from typing import Union, Dict
from girth import gum_mml
from factor_analyzer import FactorAnalyzer, ModelSpecificationParser, ConfirmatoryFactorAnalyzer
from modules.data_preparator import DataPreparator


class ScaleAnalyzer:

    def __init__(self, data_preparator: DataPreparator):
        """
        :param data_preparator: DataPreparator
            An instantiated DataPreparator class from this package
        """

        self.data_preparator = data_preparator
        self.rotation = None
        self.scale_dict = None

    def efa(self, rotation: str = 'obl', estimation: str = 'ml', n_factors: Union[str, int] = 'estimate'):
        """
        :param rotation: str
            rotation type (orthogonal or oblique)
        :param estimation: str
            Specification for estimation type (either minres or maximum likelihood)
        :param n_factors: string or int
            If string, must be 'estimate'. This will estimate eigenvalues.
            if int, the number of factors to estimate
        :return: np.array
            if n_factors == estimate, return eigenvalues, else returns factor loadings
        """

        self._efa_error_check(rotation, estimation)

        # check rotation
        if (rotation == 'orthog') or (rotation == 'varimax'):
            rotation = 'varimax'
        elif rotation == 'oblique' or (rotation == 'oblimin'):
            rotation = 'oblimin'
        else:
            raise ValueError(f"rotation parameter must be one of 'orthog' ('varimax') or 'oblique' ('oblimin'),"
                             f" but {rotation} was provided")
        if n_factors == 'estimate':
            fa = FactorAnalyzer(rotation=rotation, method=estimation)
            # access the training data through the data_preparator class
            fa.fit(self.data_preparator.x_train.values)
            # Check Eigenvalues
            ev, _ = fa.get_eigenvalues()
            return ev
        else:
            fa = FactorAnalyzer(rotation=rotation, method=estimation, n_factors=n_factors)
            fa.fit(self.data_preparator.x_train.values)
            loadings = fa.loadings_
            loadings = pd.DataFrame(loadings)
            loadings.columns = [f"factor_{i+1}" for i in range(n_factors)]
            loadings.index = self.data_preparator.x_train.columns
            # get the factor with max loading for each variables

            loadings = pd.concat([loadings, pd.Series(loadings.idxmax(axis=1), name='max_load')], axis=1)
            # Create a dictionary with factors as keys and items as values
            scale_dict = {}
            for current_factor in range(n_factors):
                factor_name = f"factor_{current_factor+1}"
                item_names = loadings[loadings['max_load'] == factor_name].index.values
                scale_dict[factor_name] = list(item_names)
            self.scale_dict = scale_dict
            return loadings

    # Not the best CFA package, but this will do for now
    def cfa(self, scale_dict: Dict):
        """
        # Not the best CFA package, but this will do for now
        :param scale_dict:
            A dictionary containing scales names as keys and a list of item names as values
        :return: output: pd.DataFrame
            return the factors loadings and factor loading errors
        """
        vars = [item for sublist in scale_dict.values() for item in sublist]
        cfa_data = self.data_preparator.x_test[vars]
        model_spec = ModelSpecificationParser.parse_model_specification_from_dict(cfa_data,
                                                                                  scale_dict)
        cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
        cfa.fit(self.data_preparator.x_test[vars].values)
        # rename columns for readability
        columns = [f"factor_{i + 1}" for i in range(len(scale_dict.keys()))]
        error_columns = [f"factor_error_{i + 1}" for i in range(len(scale_dict.keys()))]
        index = self.data_preparator.x_test[vars].columns
        loadings = pd.DataFrame(cfa.loadings_, columns=columns, index=index)
        errors = pd.DataFrame(cfa.get_standard_errors()[0], columns=error_columns, index=index)
        # combine the output
        output = pd.concat([loadings, errors], axis=1)
        return output

    def irt_gum(self, scale_dict: Dict):
        """
        :param scale_dict: Dict
            A dictionary containing scale names as keys and a list of items names as values
        :return: pd.DataFrame, pd.DataFrame
        """
        stats_results = {}
        abilities = []
        for scale_name, items in scale_dict.items():
            scale_data = np.array(self.data_preparator.data[items].astype(int)).T
            res = gum_mml(dataset=scale_data)
            des = pd.Series(res['Discrimination'], name='discrimination', index=items)
            diff = pd.DataFrame(res['Difficulties'],
                                columns=[f"difficulty_{i+1}" for i in range(0, res['Difficulties'].shape[1])],
                                index=items)
            tau = pd.DataFrame(res['Tau'], columns=[f"tau{i+1}" for i in range(0, res['Tau'].shape[1])],
                               index=items)
            ability_scores = pd.Series(res['Ability'], name=scale_name)
            # attach discrimination and Tau to the difficulties DF
            diff.insert(0, 'discrimination', des)
            out = pd.concat([diff, tau], axis=1)
            stats_results[scale_name] = out
            abilities.append(ability_scores)
        stats_results = pd.concat(stats_results, axis=0)
        abilities = pd.DataFrame(abilities).T
        abilities.columns = [f"{col}_ability" for col in abilities.columns]
        return stats_results, abilities

    def create_scales(self, scale_dict, dataset):
        """
        Create new variables representing scales
        :param scale_dict:
            A dictionary containing scales names as keys and a list of item names as values

        :return: pd.DataFrame
        """
        scales_scores = []
        for scale_name, items in scale_dict.items():
            scales_scores.append(dataset[items].mean(axis=1))
        scales_data = pd.concat(scales_scores, axis=1)
        scales_data.columns = scale_dict.keys()
        return scales_data

    def _efa_error_check(self, rotation, estimation):
        if not isinstance(rotation, str):
            raise ValueError("rotation parameter must be of type string")
        if not isinstance(estimation, str):
            raise ValueError("estimation parameter must be of type string")
        if not (estimation == 'minres') and not (estimation == 'ml'):
            raise ValueError(f"estimation parameter must be either minres or ml")

