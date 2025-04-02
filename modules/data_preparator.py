import pandas as pd
from typing import List, Union
from sklearn.model_selection import train_test_split
import category_encoders
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


class DataPreparator:

    def __init__(self, data: Union[str, pd.DataFrame], features: List[str], demo_vars: List[str] = None,
                 dep_var: str = 'target', casewise_del: bool = False, max_miss: Union[int, str, None] = None,
                 train_data: Union[str, pd.DataFrame, None] = None, val_data: Union[str, pd.DataFrame, None] = None,
                 test_data: Union[str, pd.DataFrame, None] = None):
        """
        :Parameters:
            data: pd.DataFrame
                str: A path to the location of the dataset as a string
                or
                pd.DataFrame : the raw data as a Pandas dataframe

            features: List[str]
                A list of strings containing feature names

            demo_vars: List[str]
                The names of demographic variables in the dataset for later testing (e.g., race, gender, age)

            dep_var: str
                The name of the independent variable column

            casewise_del: bool
                If True, delete cases with missing values.

            max_miss: An integer value stating the threshold for the maximium number of cases missing a variable value,
                if the number of missing is above this threshold, the entire variable is removed. This prevents reducing
                the size of the dataset when a certain value have large percentage of missing values
                IF None: no cut off (remove all cases with a missing value

            train_data: str, pd.DataFrame, or None
                A string with the path to a saved training dataset as .csv file or a pd.Dataframe containing training
                data. If none, data will be split randomly using seed when the split data method is called

            val_data: str, pd.DataFrame, or None
                A string with the path to a saved validation dataset as .csv file or a pd.Dataframe containing
                validation data. If none, data will be split randomly using seed when the split data method is called

            test_data: str, pd.DataFrame, or None
                A string with the path to a saved test dataset as .csv file or a pd.Dataframe containing test
                data. If none, data will be split randomly using seed when the split data method is called
        """
        self.features = features
        self.demo_vars = demo_vars
        self.dep_var = dep_var
        self.casewise_del = casewise_del
        self.max_miss = max_miss
        self.data = data
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.d_train = None
        self.d_val = None
        self.d_test = None
        self.import_data()

    def split_data(self, val_set: bool = False, test_size: float = .30, val_size: float = .30, random_state: int = 123,
                   shuffle: bool = True, stratify: Union[str, None] = None):
        """
        :Parameters:
            random_state: int
                A start seed for random number generation

            val_set: bool
                If true, create a validation set

            test_size: float
                A float for the percentage of data to be held as the test_set (drawn from the total sample)

            val_size: float
                A float for the percentage of data to be held as the val_set (drawn from the training sample after
                removing the test set)

            random_state : integer
                An integer for setting the seed for the random state

            shuffle: bool
                If true, shuffle data before random split

            stratify: str or None
                If string, stratify the sample using the string as column name
        """
        # ---- First, check if training, val, and test sets are provided ---- #
        if self.train_data is not None and self.test_data is not None:
            self.x_train = self.train_data[self.features]
            self.x_test = self.test_data[self.features]
            self.y_train = self.train_data[self.dep_var]
            self.y_test = self.test_data[self.dep_var]
            if self.demo_vars is not None:
                self.d_train = self.train_data[self.demo_vars]
                self.d_test = self.test_data[self.demo_vars]
            if self.val_data is not None:
                self.x_val = self.val_data[self.features]
                self.y_val = self.val_data[self.dep_var]
                if self.demo_vars is not None:
                    self.d_val = self.val_data[self.demo_vars]

        # ---- If not, automatically split the datasets ---- #
        self._split_error_checking(val_set=val_set, test_size=test_size, val_size=val_size, shuffle=shuffle,
                                   stratify=stratify)
        x = self.data[self.features]
        y = self.data[self.dep_var]
        if self.demo_vars is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
                                                                                    random_state=random_state,
                                                                                    shuffle=shuffle, stratify=stratify)
            if val_set is not None:
                self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                                      test_size=test_size,
                                                                                      random_state=random_state,
                                                                                      shuffle=shuffle,
                                                                                      stratify=stratify)
        else:
            d = self.data[self.demo_vars]
            self.x_train, self.x_test, self.y_train, self.y_test, self.d_train, self.d_test = train_test_split(x, y, d,
                                                                                               test_size=test_size,
                                                                                               random_state=random_state,
                                                                                               shuffle=shuffle,
                                                                                               stratify=stratify)
            if val_set is not None:
                self.x_train, self.x_val, self.y_train, self.y_val, self.d_train, self.d_val = train_test_split(
                                                                                    self.x_train, self.y_train,
                                                                                    self.d_train,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state,
                                                                                    shuffle=shuffle,
                                                                                    stratify=stratify)

    def impute_missing(self, strategy: str, n_neighbors: int, fill_value: Union[int, None] = -99):
        """
        :param strategy: str or None
            The strategy for handling missing data -- casewise (casewise deletion), mean, median, mode,
            constant value fill, most_frequent, iterative, or k-nearest neighbors. If None, only remove missing for
            dependent variable (target).

        :param n_neighbors: integer or None
                The value for number of nearest neighbors for KNN missing value imputation.

        :param fill_value: integer or None
                The value for constant value fill missing data imputation

        """
        missing_cols = self.data[self.features].columns[self.data[self.features].isnull().any()].tolist()

        if strategy in ['mean', 'median', 'constant', 'mode', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        elif strategy == 'iterative':
            imputer = IterativeImputer()
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
        else:
            raise ValueError(f"The strategy parameter must be one of mean, median, most_frequent, iterative, or knn, "
                             f"but {strategy} was provided")
        # fit model and transform split data
        if (self.x_train is None) or (self.x_test is None):
            raise ValueError("training or test data is missing...split data before performing missing data imputations")
        self.x_train[missing_cols] = imputer.fit_transform(self.x_train[missing_cols])
        self.x_test[missing_cols] = imputer.transform(self.x_test[missing_cols])
        if self.val_data is not None:
            self.val_data[missing_cols] = imputer.transform(self.val_data)

    def encode_categorical(self, strategy: str = 'OneHotEncoder', **kwargs):
        """
        A method to encode categorical variables. This is a wrapper around the category_encoder package.

        :param strategy: The type of categorical encoding strategy. You can choose any type from the category_encoder package.

        """
        if (self.x_train is None) or (self.x_test is None):
            raise ValueError("training or test data is missing...split data before performing categorical encoding")
        encoder_strategy = getattr(category_encoders, strategy)
        encoder = encoder_strategy(**kwargs)
        encoder.fit(X=self.x_train, y=self.y_train, cols=self.features)

        x_train = encoder.transform(self.x_train)
        x_test = encoder.transform(self.x_test)
        self.x_train = x_train
        self.x_test = x_test
        if self.x_val is not None:
            x_val = encoder.transform(self.x_val)
            self.x_val = x_val
        # Need to overwrite feature names in the case of one-hot encoding
        self.features = self.x_train.columns

    def import_data(self):
        if self.data is not None:
            raw_data = self._load_data(self.data)
            if self.demo_vars is not None:
                variables = self.features + self.demo_vars + [self.dep_var]
            else:
                variables = self.features + [self.dep_var]
            self.data = self._missing_remove(raw_data[variables])

        if (self.train_data is not None) and (self.test_data is not None):
            self.train_data = self._load_data(self.train_data)
            self.x_train = self.train_data[self.features]
            self.y_train = self.train_data[self.dep_var]

            self.test_data = self._load_data(self.test_data)
            self.x_test = self.test_data[self.features]
            self.y_test = self.test_data[self.dep_var]

            if self.demo_vars is not None:
                self.d_train = self.train_data[self.demo_vars]
                self.d_test = self.test_data[self.demo_vars]

        elif (self.train_data is not None) and (self.test_data is None):
            raise ValueError("You have manually provided a training dataset but no testing dataset. Please initiate the"
                             " DataHandler class with both datasets.")
        elif (self.train_data is None) and (self.test_data is not None):
            raise ValueError("You have manually provided a testing dataset but no training dataset. Please initiate the"
                             " DataHandler class with both datasets.")

        if self.val_data is not None:
            self.val_data = self._load_data(self.val_data)
            self.x_val = self.val_data[self.features]
            self.y_val = self.val_data[self.dep_var]

            if self.demo_vars is not None:
                self.d_val = self.val_data[self.demo_vars]

    def _load_data(self, data) -> pd.DataFrame:
        """
        Loads dataset from .csv file if required, otherwise returns the pd.DataFrame provided

        :param data:
            Eith a path to the dataset or a pd.Dataframe containing all data
        :return: pd.DataFrame
        """
        if isinstance(data, str):
            raw_data = pd.read_csv(data)
            return raw_data
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise ValueError("raw_data parameter must be either a path to a csv file or a pandas dataframe")

    def _missing_remove(self, raw_data: pd.DataFrame):
        """
        A method to remove missing data based on provided parameters.

        :param raw_data:
            A pd.DataFrame with all data

        :param max_miss:
            The maximum number of missing cases that, if reach, will cause the entire varaible to be removed.

        :param casewise: bool
            If True, remove any cases with a missing datapoint after max_miss variables are removed

        :return: pd.DataFrame with missing data removed
        """
        if self.max_miss is None:
            cleaned_data = raw_data
        elif isinstance(self.max_miss, int):
            # Keep only variables with missing counts lower than min_miss
            # get the missing value counts
            na_counts = raw_data.isna().sum()
            keep_vars = [col for col in na_counts.index if na_counts[col] < self.max_miss]
            cleaned_data = raw_data[keep_vars].dropna().reset_index(drop=True)
            # Need to change the features list now
            dropped_vars = set(raw_data.columns.values) - set(cleaned_data.columns.values)
            print(f"The following variables contain fewer than the max_miss cutoff of [{self.max_miss}]"
                  f" and were dropped from the dataset: {dropped_vars}")
            self.features = list(set(cleaned_data.columns.values) - set([self.dep_var]))
        else:
            raise ValueError(f"max_miss parameter must be either an integer value or None, but {self.max_miss} was provided")

        # for casewise deletion, remove before splitting data
        if self.casewise_del:
            cleaned_data.dropna(inplace=True).reset_index(drop=True)
        # always drop missing in dependent variable (target), never impute missing
        else:
            cleaned_data = cleaned_data.dropna(subset=[self.dep_var]).reset_index(drop=True)
        return cleaned_data

    def _split_error_checking(self, val_set, test_size, val_size, shuffle, stratify):
        if not isinstance(val_set, bool):
            raise ValueError(f"val_set parameter must be a True or False, but {type(val_set)} was provided")
        if not isinstance(test_size, float):
            raise ValueError(f"test_size parameter must be a float value, but {type(test_size)} was provided")
        if not isinstance(val_size, float):
            raise ValueError(f"val_size parameter must be a float value, but {type(val_size)} was provided")
        if not isinstance(shuffle, bool):
            raise ValueError(f"shuffle parameter must be a boolean, but {type(shuffle)} was provided")
        if not (isinstance(stratify, str)) and (stratify is not None):
            raise ValueError(f"stratify parameter must be a string value or None, but {type(stratify)} was provided")