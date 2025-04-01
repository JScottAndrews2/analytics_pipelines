from abc import ABC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR as svr, SVC as svc
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from modules.data_preparator import DataPreparator


class Model(ABC, BaseEstimator):

    def __init__(self, data_preparator: DataPreparator, **kwargs):
        # super().__init__()
        self.data_preparator = data_preparator
        self.kwargs = kwargs

    def get_params(self, deep=True):
        return {"data_preparator": self.data_preparator, **self.kwargs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, *args, **kwargs):
        self.model.fit(X=self.data_preparator.x_train, y=self.data_preparator.y_train, **kwargs)

    def predict(self, X):
        y_pred = self.model.predict(X=X)
        return y_pred


class Regressor(Model):
    def __init__(self, data_preparator):
        super().__init__(data_preparator)


class Classifier(Model, ClassifierMixin):
    def __init__(self, data_preparator):
        super().__init__(data_preparator)


class LR(Regressor):
    def __init__(self, data_preparator, **kwargs):
        super().__init__(data_preparator, **kwargs)
        self.model = LinearRegression(**kwargs)


class SVR(Regressor):
    def __init__(self, data_preparator: DataPreparator, **kwargs):
        super().__init__(data_preparator)
        self.model = make_pipeline(StandardScaler(), svr(**kwargs))


class KNN(Regressor):
    def __init__(self, data_preparator: DataPreparator, **kwargs):
        super().__init__(data_preparator)
        self.model = KNeighborsRegressor(**kwargs)


class RF(Regressor):
    def __init__(self, data_preparator: DataPreparator, **kwargs):
        super().__init__(data_preparator)
        self.model = RandomForestRegressor(**kwargs)


class MLPR(Regressor):
    def __init__(self, data_preparator: DataPreparator, **kwargs):
        super().__init__(data_preparator)
        self.model = MLPRegressor(**kwargs)


class LGR(Classifier):
    def __init__(self, data_preparator, **kwargs):
        super().__init__(data_preparator)
        self.model = LogisticRegression(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class SVC(Classifier):
    def __init__(self, data_preparator, **kwargs):
        super().__init__(data_preparator)
        self.model = svc(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class KNNC(Classifier):
    def __init__(self, data_preparator: DataPreparator, **kwargs):
        super().__init__(data_preparator)
        self.model = KNeighborsClassifier(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class RFC(Classifier):
    def __init__(self, data_preparator: DataPreparator, **kwargs):
        super().__init__(data_preparator)
        self.model = RandomForestClassifier(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class MLPC(Regressor):
    def __init__(self, data_preparator: DataPreparator, **kwargs):
        super().__init__(data_preparator)
        self.model = MLPClassifier(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
