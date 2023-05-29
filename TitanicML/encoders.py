from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class Feature_Encoder(BaseEstimator, TransformerMixin):

    def __init__(self, one_hot_features, categorical_features) -> None:
        super().__init__()
        self.one_hot_features = one_hot_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        return self
    
    def transform(self, X): 
        encoded_features = []
        for a_one_hot_f in self.one_hot_features: 
            column_data = X[a_one_hot_f].values.reshape(-1, 1)
            data = OneHotEncoder().fit_transform(column_data)
            X.drop(columns=[a_one_hot_f], inplace=True)
            column_names = [f"{a_one_hot_f}_{i}" for i in range(data.shape[1])]
            # Create a DataFrame with the one-hot encoded features
            encoded_df = pd.DataFrame(data.toarray(), columns=column_names, index=X.index)
            encoded_features.append(encoded_df)
        X = pd.concat([X]+encoded_features, axis=1)
        for a_categorical_f in self.categorical_features: 
            X[a_categorical_f] = LabelEncoder().fit_transform(X[a_categorical_f])
        return X
