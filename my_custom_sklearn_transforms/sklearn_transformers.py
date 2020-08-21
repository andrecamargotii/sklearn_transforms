from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing

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

class NormFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        data = X.copy()
        scaler = StandardScaler()
        return scaler.fit_transform(data)
