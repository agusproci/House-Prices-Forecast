import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import date

class VariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Compute age from YearBuilt
        X.loc[:, 'YearBuilt'] = date.today().year - X['YearBuilt'].values
        # Sum ground living area and basement area
        X.loc[:, 'GrLivArea'] = X['GrLivArea'] + X['TotalBsmtSF']
        # Drop original basement area column
        X = X.drop(['TotalBsmtSF'], axis=1)
        # Log-transform the sale price
        X['SalePrice'] = X['SalePrice'].apply(np.log)
        return X
