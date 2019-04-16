#!/usr/bin/env python -W ignore::DeprecationWarning
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from skgarden import RandomForestQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
import abc
import warnings
warnings.filterwarnings("ignore")


class QuantileRegression:
    __metaclass__ = abc.ABCMeta

    def __init__(self, qt, x, y, params={}):
        """
        Parameters
        ----------
        qt: float
            the quantile we want to estimate
        x: DataFrame
            feature dataset
        y: DataFrame
            target dataset
        params: dictionary
            a dictionary containing hyper-parameter key-value pairs of the model
        """
        self.qt = qt
        self.x = x
        self.y = y
        self.params = params

    @abc.abstractmethod
    def fit_model(self):
        pass

    @abc.abstractmethod
    def feature_importance(self):
        pass

    @abc.abstractmethod
    def predict(self, data):
        pass

    def __score(self, y_true, y_pred):
        """
        Parameters
        ----------
        y_true: ndarray
            a numpy array of actual values.
        y_pred: ndarray
            a numpy array of predicted values.

        Return
        -------------------
        output : float
            the quantile score of predicted values.

        Notes
        -----
        The quantile score is computed by solving the equation:
            QS = mean(qt * max(resid, 0)+(1 - qt)*max(-resid, 0))
        """
        resid = y_true - y_pred
        return np.mean(resid * np.where(resid > 0, self.qt, self.qt - 1))

    def loss_score(self):
        """
        Calculate loss function scores for the whole data set

        Returns
        -------
        output: float
            quantile loss function score for the train dataset
        """
        return self.__score(self.y, self.predict(self.x))


class LinearQuantileRegression(QuantileRegression):
    def __init__(self, qt, x, y, params={}):
        """
        Parameters
        ----------
        qt: float
            the quantile we want to estimate
        x: DataFrame
            feature dataset
        y: DataFrame
            target dataset
        params: dictionary
            a dictionary containing hyper-parameter key-value pairs of the model

        Internal Attributes
        -------------------
        self.linear_quantile : statsmodels.regression.linear_model.RegressionResultsWrapper object or None
          the fitted model

        """
        super(LinearQuantileRegression, self).__init__(qt, x, y, params)
        self.linear_quantile = None
        self.fit_model()

    def fit_model(self):
        """
        fit the linear quantile regression model using the train dataset

        Returns
        -------
        output: statsmodels.regression.linear_model.RegressionResultsWrapper object
            the linear quantile regression model
        """
        x_columns = list(self.x.columns.values)
        equation = self.y.name + '~' + '+'.join(x_columns)
        df = pd.concat([self.y, self.x], axis=1)
        self.linear_quantile = smf.quantreg(equation, data=df)
        self.linear_quantile = self.linear_quantile.fit(q=self.qt)
        return self.linear_quantile

    def feature_importance(self):
        """
        Sort the features from most important to least important

        Returns
        -------
        output: Series
            Sort the features from most important to least important with corresponding p-values
        """
        ordered_pvalues = self.linear_quantile.pvalues.sort_values(ascending=True)
        ordered_pvalues.name = 'p-values'
        ordered_pvalues = ordered_pvalues.drop(labels='Intercept')  # we don't care about the intercept term
        return ordered_pvalues

    def predict(self, data):
        """
        predict the qt th quantile for data

        Parameters
        ----------
        data: DataFrame
             new data

        Returns
        -------
        output: numpy.ndarray
            predicted quantile for data
        """

        return np.asarray(self.linear_quantile.predict(sm.add_constant(data)))


class RandomForestRegression(QuantileRegression):
    def __init__(self, qt, x, y, params={}):
        """
        Parameters
        ----------
        qt: float
            the quantile we want to estimate
        x: DataFrame
            feature dataset
        y: DataFrame
            target dataset
        params: dictionary
            a dictionary containing hyper-parameter key-value pairs of the model

        Internal Attributes
        -------------------
        self.random_forest : RandomForestQuantileRegressor Object or None
          the fitted model
        """
        super(RandomForestRegression, self).__init__(qt, x, y, params)
        self.random_forest = None
        self.fit_model()

    def fit_model(self):
        """
        fit the gradient boosting regression model using the train dataset

        Returns
        -------
        output: RandomForestQuantileRegressor object
            the random forest quantile regression model
        """
        x_train_dummy = pd.get_dummies(self.x)
        self.random_forest = RandomForestQuantileRegressor()
        self.random_forest.set_params(**self.params)
        self.random_forest = self.random_forest.fit(x_train_dummy, self.y)
        return self.random_forest

    def feature_importance(self):
        """
        Sort the features from the most important to the least important

        Returns
        -------
        output: Series
            Sort the features from the most important to the least important with corresponding values
        """
        feature_importances = self.random_forest.feature_importances_
        feature_importances = pd.Series(feature_importances, index=pd.get_dummies(self.x).columns)
        return feature_importances.sort_values(ascending=False)

    def predict(self, data):
        """
        predict the qt th quantile for new data

        Parameters
        ----------
        data: DataFrame
             new data

        Returns
        -------
        output: numpy.ndarray
            predicted quantile for new data
        """
        data_dummy = pd.get_dummies(data)
        return self.random_forest.predict(data_dummy, quantile=self.qt * 100)


class GradientBoostingRegression(QuantileRegression):
    def __init__(self, qt, x, y, params={}):
        """
        Parameters
        ----------
        qt: float
            the quantile we want to estimate
        x: DataFrame
            feature dataset
        y: DataFrame
            target dataset
        params: dictionary
            a dictionary containing hyper-parameter key-value pairs of the model

        Internal Attributes
        -------------------
        self.gradient_boosting : GradientBoostingRegressor Object or None
          the fitted model
        """
        super(GradientBoostingRegression, self).__init__(qt, x, y, params)
        self.gradient_boosting = None
        self.fit_model()

    def fit_model(self):
        """
        fit the gradient boosting regression model using the train dataset

        Returns
        -------
        output: GradientBoostingRegressor Object
            the gradient boosting regression model

        Notes
        -----
        by setting loss function equal to "quantile" and alpha equal to qt,
        we can estimate qt th quantile of data
        """
        x_train_dummy = pd.get_dummies(self.x)
        self.gradient_boosting = GradientBoostingRegressor(loss='quantile', alpha=self.qt)
        self.gradient_boosting.set_params(**self.params)
        self.gradient_boosting = self.gradient_boosting.fit(x_train_dummy, self.y)
        return self.gradient_boosting

    def feature_importance(self):
        """
        Sort the features from the most important to the least important

        Returns
        -------
        output: Series
            Sort the features from most important to least important with corresponding values
        """
        feature_importances = self.gradient_boosting.feature_importances_
        feature_importances = pd.Series(feature_importances, index=pd.get_dummies(self.x).columns)
        return feature_importances.sort_values(ascending=False)

    def predict(self, data):
        """
        predict the qt th quantile for new data

        Parameters
        ----------
        data: DataFrame
             new data

        Returns
        -------
        output: numpy.ndarray
            predicted quantile for new data
        """
        data_dummy = pd.get_dummies(data)
        return self.gradient_boosting.predict(data_dummy)
