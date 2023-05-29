from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin



class Feature_Encoder(BaseEstimator, TransformerMixin):

    def __init__(self, one_hot_features, categorical_features) -> None:
        super().__init__()
        self.one_hot_features = one_hot_features
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        self.encoders = []
        for a_one_hot_f in self.one_hot_features: 
            self.encoders.append((a_one_hot_f, LabelEncoder.fit(X[a_one_hot_f])))
        for a_categorical_f in self.categorical_features: 
            self.encoders.append((a_categorical_f, OneHotEncoder.fit(X[a_categorical_f])))
        return self
    
    def transform(self, X): 
        for a_feature, a_encoder in self.encoders:
            X[a_feature] = a_encoder.fit(X[a_feature])
        return X
