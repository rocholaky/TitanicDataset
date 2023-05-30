from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class Feature_Encoder(BaseEstimator, TransformerMixin):

    def __init__(self, one_hot_features, categorical_features) -> None:
        super().__init__()
        self.one_hot_features = one_hot_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        self.one_hot_encoders = []
        self.label_encoders = []
        for a_one_hot_f in self.one_hot_features: 
            column_data = X[a_one_hot_f].unique().tolist()
            self.one_hot_encoders.append(OneHotEncoder().fit(column_data))
        for a_categorical_f in self.categorical_features: 
             column_data = X[a_categorical_f].unique().tolist()
             self.label_encoders.append(LabelEncoder().fit(column_data))
        return self
    
    def transform(self, X): 
        encoded_features = []
        for a_one_hot_f, a_one_hot_encoder in zip(self.one_hot_features, self.one_hot_encoders): 
            column_data = X[a_one_hot_f].values.reshape(-1, 1)
            data = a_one_hot_encoder.transform(column_data)
            X.drop(columns=[a_one_hot_f], inplace=True)
            column_names = [f"{a_one_hot_f}_{i}" for i in range(data.shape[1])]
            # Create a DataFrame with the one-hot encoded features
            encoded_df = pd.DataFrame(data.toarray(), columns=column_names, index=X.index)
            encoded_features.append(encoded_df)
        X = pd.concat([X]+encoded_features, axis=1)
        for a_categorical_f, a_label_encoder in zip(self.categorical_features, self.label_encoders): 
            X[a_categorical_f] = a_label_encoder.transform(X[a_categorical_f])
        return X
