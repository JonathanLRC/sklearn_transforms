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


class ReplaceObjects(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        for c in data.columns:
            if data[c].dtype == 'object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(data[c].values))
                data[c] = lbl.transform(list(data[c].values))
        # Devolvemos un nuevo dataframe de datos con los objetos substituidos
        return data

