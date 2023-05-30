from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import re



class NameTransformer(BaseEstimator, TransformerMixin): 
    def Analyze_titles(self, x): 
        # Function that returns 1 if the titles are Mr. or Mrs. which are titles normaly
        # given to adults and 0 otherwise. 
        pattern= "(?<= )Mr|Mrs|Dr|Rev|Sir(?=[.])"
        search = re.search(pattern, x["Name"])
        if search: 
            return 1
        else: 
            return 0
        
    def analyze_men_marriage(self, passenger, grouped_families_list): 
        grouped_families_list = grouped_families_list[grouped_families_list.index != passenger.name]
        has_wife = np.logical_and(grouped_families_list["isAdult"],\
                             (grouped_families_list["Sex"]=="female"),
                            grouped_families_list["isMarried"])
        justSp = passenger["SibSp"] == 1
        return any(has_wife) * justSp

    def Generate_marriage(self, df):
        df["isMarried"] = df.apply(lambda x: int("(" in x["Name"]), axis=1)
        married_candidates = df[(df["Sex"]=="male") & (df["isAdult"])]
        are_married = married_candidates.apply(
                        lambda x: self.analyze_men_marriage(
                            x, df.loc[df["familyId"].eq(x["familyId"])]
                        ),axis=1)
        df.loc[married_candidates.index, "isMarried"]= are_married
        return df

    def fit(self, X, y=None): 
        
        return self
    
    def transform(self, X): 
        # generate new column called LastName:
        self.family_last_name = X.apply(lambda x: str((x["Name"].split(",")[0], x["Ticket"])), axis=1)
        X["familyId"] = self.family_last_name
        self.LastNameGroups = X.groupby(["familyId"])
        X["isAdult"] = X.apply(lambda x: int(self.Analyze_titles(x) or x["Age"]>=18), axis=1)
        X = self.Generate_marriage(X)
        return X


class Family_size:
    key = "Family_size"
    
    def __call__(self, df):
       df["familySize"] = df.apply(lambda x: x["SibSp"]+x["Parch"]+1, axis=1)
       return df
    
class isAlone:
    def __call__(self, df):
        df["isAlone"] = df.apply(lambda x: int(x["familySize"]==1), axis=1)
        return df
    

class SimpleFeatureGeneration(BaseEstimator, TransformerMixin): 
    features_2_generate= [Family_size, isAlone]
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X): 
        for a_feature in self.features_2_generate:
            X = a_feature()(X)
        return X
    

class FeatureDropper(BaseEstimator, TransformerMixin): 
    features_2_drop = ["familyId", "Cabin", "Ticket", "Name"]
    
    def fit(self, X, y=None): 
        return self
    
    def transform(self, X): 
        X.drop(columns=self.features_2_drop, inplace=True)
        return X


