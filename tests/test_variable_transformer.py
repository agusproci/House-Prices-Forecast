import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from datetime import date
from variable_transformer import VariableTransformer


def test_yearbuilt_transformation():
    df = pd.DataFrame({
        'YearBuilt': [2000, 2010],
        'GrLivArea': [1000, 1500],
        'TotalBsmtSF': [500, 700],
        'SalePrice': [200000, 250000],
    })
    vt = VariableTransformer()
    transformed = vt.fit_transform(df)
    expected = [date.today().year - 2000, date.today().year - 2010]
    assert list(transformed['YearBuilt']) == expected
