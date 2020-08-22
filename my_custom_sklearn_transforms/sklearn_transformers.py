from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class ReplaceDfNaN(BaseEstimator, TransformerMixin):
    def __init__(self, value=0):
        self.nan = value
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        data = X.copy()
        return data.fillna(self.nan)

class NewFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        data = X.copy()
        data['MEDIA'] =  data.iloc[:, 0:4].mean(axis=1)
        data['RENDIMENTO'] = (data <= 6).astype(int).sum(axis=1)
        return data
    
class NormFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self.feature = feature
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        data = X.copy()
        data[data[1:4] > 10] = 10
        return data
