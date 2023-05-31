from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer



class AgeHandler(BaseEstimator, TransformerMixin): 
    AdultAge = None
    ChildrenAge = None

    def fit(self, X, y=None):
        self.adult_age_mean = X[X["isAdult"]==1]["Age"].mean()
        self.children_age_mean = X[X["isAdult"]==0]["Age"].mean()
        return self
    
    def transform(self, X): 
        X.loc[(X["isAdult"]==1) & (X["Age"].isna()), "Age"]= self.adult_age_mean
        X.loc[(X["isAdult"]==0) & (X["Age"].isna()), "Age"] = self.children_age_mean
        return X


class EmbarkedHandler(BaseEstimator, TransformerMixin): 
    def fit(self, X, y=None): 
        embarked_index = X[X["Embarked"].isna()].index
        X.drop(index= embarked_index, inplace=True)
        y.drop(index=embarked_index, inplace=True)
        return self
    
    def transform(self, X): 
        return X
    


