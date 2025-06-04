#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[314]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor

from statsmodels.stats.outliers_influence import variance_inflation_factor
from datetime import date




# In[363]:


housing = pd.read_csv('./input/train.csv',index_col=0).reset_index(drop=True)


# ## Take a Quick Look at the Data Structure

# In[316]:


housing.head()


# In[317]:


housing.info()


# In[318]:


#Check null values in each features with null values
sns.set(rc = {'figure.figsize':(10,10)})
sns.heatmap(housing[housing.columns[housing.isna().any()].tolist()].isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[319]:


housing[housing.columns[housing.isna().any()].tolist()].info()


# In[320]:


#descriptive statistics summary
housing['SalePrice'].describe()


# In[321]:


#histogram
sns.distplot(housing['SalePrice']);


# In[322]:


#skewness and kurtosis
print("Skewness: %f" % housing['SalePrice'].skew())
print("Kurtosis: %f" % housing['SalePrice'].kurt())


# In[323]:


#histogram of log(SalePrice)
sns.distplot(housing['SalePrice'].apply(np.log));


# In[324]:


#skewness and kurtosis of log transformed data
print("Skewness: %f" % housing['SalePrice'].apply(np.log).skew())
print("Kurtosis: %f" % housing['SalePrice'].apply(np.log).kurt())


# In[325]:


corr_matrix = housing.corr(method='spearman', numeric_only=True).abs().sort_values(by='SalePrice', ascending=False)
cor_target = abs(corr_matrix["SalePrice"])
#Selecting highly correlated features
num_var = cor_target[cor_target>0.4]
#target variable
target = num_var['SalePrice']
relevant_features = num_var.drop(['SalePrice'])
num_var


# In[326]:


#correlation matrix
housing_num = housing[num_var.index]
# Transform the target variable to log scale
housing_num['SalePrice'] = housing_num['SalePrice'].apply(np.log)
corrmat = housing_num.corr(method='spearman', numeric_only=True).abs()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True, annot=True);


# In[327]:


# Create correlation matrix
housing_num_feat = housing[relevant_features.index]
corr_matrix = housing_num_feat.corr(method='spearman', numeric_only=True).abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
to_drop


# In[328]:


# Drop highly correlated features
# Drop garage area as it is highly correlated with garage cars and have lower correlation with SalePrice
housing_num.drop(['GarageArea'], axis=1, inplace=True)
# Drop garage year built as it is highly correlated with year built and have lower correlation with SalePrice
housing_num.drop(['GarageYrBlt'], axis=1, inplace=True)
# Drop First Floor square feet as it is highly correlated with Total square feet of basement area and have lower correlation with SalePrice
housing_num.drop(['1stFlrSF'], axis=1, inplace=True)
# Drop Total rooms above grade as it is highly correlated with TAbove grade (ground) living area square feet and have lower correlation with SalePrice
housing_num.drop(['TotRmsAbvGrd'], axis=1, inplace=True)


# In[329]:


corrmat = housing_num.corr(method='spearman', numeric_only=True).abs()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True, annot=True);


# In[330]:


# Impute the missing values with the KNN algorithm
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")
imputer.fit_transform(housing_num)
# Create dataframe with data from imputer
housing_num = pd.DataFrame(imputer.transform(housing_num), columns=housing_num.columns)


# In[331]:


housing_num.describe(include='all')


# In[332]:


# identify outliers in the dataset
clf = LocalOutlierFactor(n_neighbors=2)
clf.fit_predict(housing_num)
# count the number of outliers in the dataset
outliers = pd.Series(clf.negative_outlier_factor_, index=housing_num.index)
# remove the outliers from the dataset  
housing_num = housing_num[clf.negative_outlier_factor_ > -2.5]


# In[333]:


housing_num


# In[334]:


housing_num.describe(include='all')


# In[335]:


# Calculte the mutual information scores for each feature
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y):
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

X_train = housing_num.loc[:, housing_num.columns != 'SalePrice']
y_train=housing_num['SalePrice']
mi_scores = make_mi_scores(X_train, y_train)
mi_scores[::3]  # show a few features with their MI scores


# In[336]:


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)


# In[337]:


#Compute VIF data for each independent variable


features = housing_num.drop(['SalePrice'], axis=1)

vif = pd.DataFrame()
vif["features"] = features.columns
vif["vif_Factor"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
vif


# In[338]:


# Transforming the Year build to years old and dropping the original column and the year remodelled as it is highly correlated with year built
features = housing_num.drop(['SalePrice'], axis=1)
# Calculate the age of the house in years. Using the current year minus the
# year the house was built ensures a positive number of years.
features['YearOld'] = date.today().year - features['YearBuilt']
features['TotGrArea'] = features['GrLivArea'] + features['TotalBsmtSF']
# Dropping the LotFrontage as it is highly correlated with LotArea and have lower correlation with SalePrice
features = features.drop(['YearBuilt','YearRemodAdd','LotFrontage','GrLivArea','TotalBsmtSF'], axis=1)

vif = pd.DataFrame()
vif["features"] = features.columns
vif["vif_Factor"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
vif


# In[339]:


features['SalePrice'] = housing_num['SalePrice']
corrmat = features.corr(method='spearman', numeric_only=True).abs()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True, annot=True);


# In[340]:


# Removing variables in housing_num
# Likewise update the numerical dataframe.
housing_num['YearOld'] = date.today().year - housing_num['YearBuilt']
housing_num['TotGrArea'] = housing_num['GrLivArea'] + housing_num['TotalBsmtSF']
housing_num = housing_num.drop(['YearBuilt','YearRemodAdd','LotFrontage','GrLivArea','TotalBsmtSF'], axis=1)


# In[341]:


# Get the categorical features
housing_cat = housing.select_dtypes(include=['object'])
housing_cat.head()


# In[342]:


housing_cat.describe(include='all')


# In[343]:


housing_cat.isna().sum()


# In[344]:


# Impute the missing values with None from data description
housing_cat = housing_cat.fillna('None')


# In[345]:


housing_cat.columns


# In[346]:


# Encode the categorical features into numerical features with ordinal encoding
from sklearn.preprocessing import OrdinalEncoder
# Columns to encode
colums_ord= ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC','BsmtFinType1','BsmtFinType2','Functional','GarageFinish','Fence','LandSlope','LotShape','PavedDrive']
ordinal_encoder = OrdinalEncoder()
housing_cat_ord = housing_cat[colums_ord]
housing_cat_encoded_ord = ordinal_encoder.fit_transform(housing_cat_ord)
housing_cat_encoded_ord = pd.DataFrame(housing_cat_encoded_ord, columns=housing_cat_ord.columns)


# In[347]:


housing_cat.drop(colums_ord, axis=1, inplace=True)
housing_cat=pd.concat([housing_cat, housing_cat_encoded_ord], axis=1)


# In[348]:


housing_cat


# In[349]:


# select rest of the columns to encode type object
columns_one = housing_cat.select_dtypes(include=['object']).columns
# Encode the categorical features into numerical features with one hot encoding
one_hot_encoder = OneHotEncoder()
housing_cat_one = housing_cat[columns_one]
housing_cat_encoded_one = one_hot_encoder.fit_transform(housing_cat_one)
housing_cat_encoded_one = pd.DataFrame(housing_cat_encoded_one.toarray(), columns=one_hot_encoder.get_feature_names_out(columns_one))
# concat the encoded columns with the original dataframe
housing_cat.drop(columns_one, axis=1, inplace=True)
housing_cat_encoded=pd.concat([housing_cat, housing_cat_encoded_one], axis=1)


# In[350]:


housing_cat


# In[351]:


housing_cat_encoded


# In[352]:


# Add the sale price to the encoded categorical features
housing_cat_encoded['SalePrice'] = housing_num['SalePrice']
# remove rows with missing values
housing_cat_encoded = housing_cat_encoded.dropna()
housing_cat_encoded


# In[353]:


# Select the best features with random forest


select = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold=-np.inf, max_features=25)
select.fit(housing_cat_encoded.drop('SalePrice',axis=1), housing_cat_encoded['SalePrice'])

# Get the selected features
selected_features = select.get_support()
selected_features_names = housing_cat_encoded.drop('SalePrice',axis=1).columns[selected_features]



# In[354]:


selected_features_names


# In[355]:


# Plot the feature importance
feature_importances = pd.Series(select.estimator_.feature_importances_, index=housing_cat_encoded.drop('SalePrice',axis=1).columns)
feature_importances.nlargest(25).plot(kind='barh', figsize=(10,10));


# In[356]:


housing_cat = housing_cat_encoded[selected_features_names]
housing_cat['SalePrice'] = housing_cat_encoded['SalePrice']


# In[357]:


# Concatenate the numerical and categorical features
housing_transformed = pd.concat([housing_num, housing_cat], axis=1)
housing_transformed.reset_index(drop=True, inplace=True)
housing_transformed = housing_transformed.loc[:,~housing_transformed.columns.duplicated()].copy()


# In[358]:


housing_transformed


# In[359]:


housing_transformed.columns


# In[361]:


# The final features in housing_transformed are:
num_feat = ['SalePrice', 'OverallQual', 'GarageCars', 'FullBath', 'Fireplaces','OpenPorchSF', 'LotArea', 'MasVnrArea', 'YearBuilt', 'GrLivArea','TotalBsmtSF']
cat_feat_one_hot = ['MSZoning','Neighborhood','GarageType','CentralAir','HouseStyle','BsmtExposure']
cat_feat_ord = ['ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu','GarageQual', 'GarageCond', 'BsmtFinType1', 'GarageFinish', 'LotShape','PavedDrive']


# In[364]:


# Create custom transformer for the numerical features

class VariableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self): # no *args or **kargs
        pass
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        # Create new feature Year Old
        X.loc[:,'YearBuilt'] = date.today().year - X['YearBuilt'].values
        # Create new feature TotGrArea
        X.loc[:,'GrLivArea'] = X.loc[:,'GrLivArea'] + X.loc[:,'TotalBsmtSF']
        # Drop the other features
        X = X.drop(['TotalBsmtSF'], axis=1)
        # Transform sale price to log scale
        X['SalePrice'] = X['SalePrice'].apply(np.log)
        return X.values


# In[365]:


# Create another transformer for outlier detection with LOF

class OutlierRemoval(BaseEstimator, TransformerMixin):
    def __init__(self): # no *args or **kargs
        pass
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        # create new feature for outlier detection
        # remove the outliers with LOF
        # return the dataframe without the outliers
        # return X.values
        lof = LocalOutlierFactor(n_neighbors=2)
        lof.fit(X)
        outliers = lof.negative_outlier_factor_ < -2.5
        # set the outliers to nan
        X[outliers] = np.nan
        return X


# In[367]:


# create pipeline for the numeric data

num_pipeline = Pipeline([
        ('attribs_adder', VariableTransformer()),
        ('imputer', KNNImputer(n_neighbors=2, weights="uniform")),
        ('outlier_remover', OutlierRemoval()),

    ])


# In[370]:


# create categorical pipeline for one hot encoding

cat_pipeline_one = Pipeline([
        ('imputer_cat', SimpleImputer(strategy="constant", fill_value="None")),
        ('one_hot_encoding', OneHotEncoder(sparse=False)),

    ])


# In[371]:


# create categorical pipeline for ordinal encoding

cat_pipeline_ord = Pipeline([
        ('imputer_cat', SimpleImputer(strategy="constant", fill_value="None")),
        ('ord_encoding', OrdinalEncoder()),

    ])


# In[372]:


# Create a full pipeline for the numerical and categorical features

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_feat),
        ("cat one", cat_pipeline_one, cat_feat_one_hot),
        ("cat ord", cat_pipeline_ord, cat_feat_ord),
    ])
# Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
# Drop nan values where the outliers were stored
housing_prepared = housing_prepared[~np.isnan(housing_prepared).any(axis=1)]


# In[379]:


# Split the data into train and test set

X_train, X_test, y_train, y_test = train_test_split(housing_prepared[:,1:], housing_prepared[:,0], test_size=0.3, random_state=42)


# In[380]:


# Scale the data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[381]:


# Create a random forest model with gridsearchCV


param = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'max_features': ['auto', 'sqrt', 'log2']}
forest_reg = RandomForestRegressor()
grid_search_rf = GridSearchCV(forest_reg, param, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print(grid_search_rf.best_params_)
print("Error Training Set: ",np.exp(-grid_search_rf.best_score_))
print("Error Test Set: ", np.exp(-grid_search_rf.score(X_test, y_test)))




# In[382]:


# Create a XGBoost model with gridsearchCV


param = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
xgb_reg = XGBRegressor()
grid_search_xg = GridSearchCV(xgb_reg, param, cv=5, scoring='neg_mean_squared_error')
grid_search_xg.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print(grid_search_xg.best_params_)
print("Error Training Set: ",np.exp(-grid_search_xg.best_score_))
print("Error Test Set: ", np.exp(-grid_search_xg.score(X_test, y_test)))


# In[383]:


# Create a Lasso model with gridsearchCV


param = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
lasso_reg = Lasso()
grid_search_lasso = GridSearchCV(lasso_reg, param, cv=5, scoring='neg_mean_squared_error')
grid_search_lasso.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print(grid_search_lasso.best_params_)
print("Error Training Set: ",np.exp(-grid_search_lasso.best_score_))
print("Error Test Set: ", np.exp(-grid_search_lasso.score(X_test, y_test)))


# In[390]:


# Create a Ridge model with gridsearchCV



param = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
ridge_reg = Ridge()
grid_search_ridge = GridSearchCV(ridge_reg, param, cv=5, scoring='neg_mean_squared_error')
grid_search_ridge.fit(X_train, y_train)

# Print the best parameters and the corresponding score

print(grid_search_ridge.best_params_)
print("Error Training Set: ",np.exp(-grid_search_ridge.best_score_))
print("Error Test Set: ", np.exp(-grid_search_ridge.score(X_test, y_test)))

