from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from abc import ABC
from sklearn.pipeline import Pipeline, make_pipeline
from TitanicML.missingDataHandlers import *
from TitanicML.encoders import *
from TitanicML.transformer import *
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

class cliModelBase(ABC): 
    model_base= None
    classifier_name = None
    help_text = None
    param_dict = None
    selected_parameters = {}
    key = None
    
    def get_parameters(self): 
        for a_key in  self.param_dict.keys():
            self.key = a_key
            yield a_key
    
    def show_option(self, key): 
        help_text = self.param_dict.get(key).get("help text")
        type_ = self.param_dict.get(key).get("type")
        text = f"{key} ({type_}): {help_text},\n"
        if type_ is list: 
            options = self.param_dict.get(key).get("options")
            for i, a_option in enumerate(options): 
                text += f"{i+1}. {a_option}\n"
            default = self.param_dict.get(key).get("default")
            text += f"default:{default}\n"
        else: 
            min_ = self.param_dict.get(key).get("min")
            max_ = self.param_dict.get(key).get("max", "no max")
            default = self.param_dict.get(key).get("default")
            text += f"min={min_},\nmax={max_},\ndefault={default}\n"
        return text
    
    
    def generate_model(self):
        if len(self.selected_parameters)==0:
            model =  self.model_base()
        else:
            model =  self.model_base(**self.selected_parameters)
        random_state = 42
        model.random_state = random_state
        return model

    def numeric_validate_choice(self, value): 
        parameter_dict = self.param_dict
        selected_param = self.param_dict[self.key]
        if value< selected_param["min"]:  
            raise ValueError("The value is lower than the minimum")
        elif "max" in parameter_dict.keys():
            if value>parameter_dict["max"]: 
                raise ValueError("The value is Bigger than the maximum value")
        
    def str_validate_choice(self, value):
        selected_param = self.param_dict[self.key]
        if not value in selected_param["options"]: 
            raise ValueError("Choice is not an option! Please choose one of the following"+\
                             ",".join(selected_param["options"]))
        
    def None_validate_choice(self): 
        selected_param = self.param_dict[self.key]
        if not selected_param["default"] is None: 
            raise ValueError("None Value cant be selected here")

    def select_choice(self, option): 
        type_ = self.param_dict.get(self.key).get("type")
        if type_ is list: 
            selection_type = str
        else: 
            selection_type = type_
        try: 
            value = selection_type(option) if option else option
            if selection_type is str: 
                self.str_validate_choice(value)
            elif selection_type is int and value is None:
                self.None_validate_choice()
            else: 
                self.numeric_validate_choice(value)
        except: 
            raise ValueError(f"Wrong Value Given! the type of value should be {selection_type}")
        else: 
            self.selected_parameters[self.key] = value
            self.key=None

    def generate_grid(self): 
        grid = {}
        for a_param, value in self.param_dict.items():
            type_ = value["type"]
            if type_ is list: 
                grid[a_param] = tuple(value["options"])
            else: 
                min_ = value.get("min", 0)
                default = value["default"] if not value["default"] is None else 2*(min_+1)
                max_ = value.get("max", 2*default)
                n_search = value.get("n_search")
                if type_ is int: 
                    grid[a_param] = list(set([default] + np.linspace(start=min_, stop=min(max_, 2*default), num=n_search, dtype=int).tolist()))
                else: 
                    grid[a_param] = list(set([default] + np.linspace(start=min_, stop=min(max_, 10*default), num=n_search, dtype=float).tolist()))

        return grid




class SVM(cliModelBase): 
    model_base = SVC
    classifier_name = "SVM"
    help_text = "SVM (Support Vector Machine) is a powerful machine learning algorithm used for classification and regression tasks.\n\
                 It finds an optimal hyperplane to separate different classes in the data by maximizing the margin between them.\n \
                SVM is effective for handling high-dimensional data and can handle both linear and non-linear relationships through\n \
                kernel functions.\n"
    param_dict = {"C": {"help text": "regularization term", 
                        "type": float,
                        "min": 0.01,
                        "default":1, 
                        "n_search": 4},
                    "kernel": {"help text": "type of kernel to use in order to apply the kernel Trick", 
                               "type": list, 
                               "options": ["linear", "poly", "rbf", "sigmoid"],
                               "default": "rbf"},
                    "degree": {"help text": "polynomial Degree of kernels (ONLY FOR POLY)", 
                               "type": int,
                               "min": 0,
                               "default": 3,
                               "n_search": 2}}
    
    def __init__(self):
        super().__init__()
        self.selected_parameters = {}
    
class RandomForest(cliModelBase):
    model_base = RandomForestClassifier
    classifier_name = "Random Forest"
    help_text = "Random Forest is an ensemble machine learning algorithm that uses multiple decision trees to make predictions.\n\
    It combines the predictions of individual trees to produce a final prediction. Random Forest is effective\n\
    for both classification and regression tasks. It handles high-dimensional data and can capture complex\n\
    relationships between features. It also provides measures of feature importance."
    param_dict = {
        "n_estimators": {
            "help text": "number of trees in the forest",
            "type": int,
            "min": 1,
            "default": 100,
            "n_search": 3
        },
        "criterion": {
            "help text": "function to measure the quality of a split",
            "type": list,
            "options": ["gini", "entropy"],
            "default": "gini",
            "n_search": 1
        },
        "max_depth": {
            "help text": "maximum depth of the trees",
            "type": int,
            "min": 1,
            "default": None,
            "n_search": 2
        },
        "min_samples_split": {
            "help text": "minimum number of samples required to split an internal node",
            "type": int,
            "min": 2,
            "default": 2,
            "n_search": 2
        },
        "min_samples_leaf": {
            "help text": "minimum number of samples required to be at a leaf node",
            "type": int,
            "min": 1,
            "default": 1,
            "n_search": 1
        }
    }

    def __init__(self):
        super().__init__()
        self.selected_parameters = {}


class NaiveBayes(cliModelBase):
    model_base = GaussianNB
    classifier_name = "Naive Bayes"
    help_text = "Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with an assumption\n\
    of independence between features. It is commonly used for binary classification problems and\n\
    works well with categorical or numerical features. Naive Bayes is fast and efficient, making\n\
    it suitable for large datasets."
    param_dict = {
        "var_smoothing": {
            "help text": "portion of the largest variance of all features added to\n\
                            variances for calculation stability",
            "type": float,
            "min": 1e-9,
            "default": 1e-9,
            "n_search": 1
        }
    }

    def __init__(self):
        super().__init__()
        self.selected_parameters = {}


class XGBoost(cliModelBase):
    model_base = xgb.XGBClassifier
    classifier_name = "XGBoost"
    help_text = "XGBoost (Extreme Gradient Boosting) is a powerful gradient boosting framework widely used for\n\
    classification and regression tasks. It employs a gradient boosting algorithm that iteratively\n\
    combines weak learners (decision trees) to form a strong predictive model. XGBoost is known\n\
    for its high performance, scalability, and ability to handle complex datasets with large\n\
    numbers of features."
    param_dict = {
        "max_depth": {
            "help text": "maximum depth of a tree",
            "type": int,
            "min": 1,
            "default": 3,
            "n_search": 3
        },
        "learning_rate": {
            "help text": "step size shrinkage used to prevent overfitting",
            "type": float,
            "min": 0.01,
            "default": 0.1,
            "n_search": 2
        },
        "n_estimators": {
            "help text": "number of boosted trees to fit",
            "type": int,
            "min": 1,
            "default": 100,
            "n_search": 3
        },
        "subsample": {
            "help text": "subsample ratio of the training instances",
            "type": float,
            "min": 0.01,
            "max": 1,
            "default": 1,
            "n_search": 2
        },
        "colsample_bytree": {
            "help text": "subsample ratio of columns when constructing each tree",
            "type": float,
            "min": 0.01,
            "max": 1,
            "default": 1,
            "n_search": 2
        },
        "reg_alpha": {
            "help text": "L1 regularization term on weights",
            "type": float,
            "min": 0,
            "default": 0,
            "n_search": 1
        },
        "reg_lambda": {
            "help text": "L2 regularization term on weights",
            "type": float,
            "min": 0,
            "default": 1,
            "n_search": 1
        }
    }

    def __init__(self):
        super().__init__()
        self.selected_parameters = {}


class DecisionTree(cliModelBase):
    model_base = DecisionTreeClassifier
    classifier_name = "Decision Tree"
    help_text = "Decision Tree is a versatile and widely used classification algorithm that builds a tree-like model\n\
    of decisions based on features and their thresholds. It splits the data based on the feature that\n\
    provides the most information gain or Gini impurity reduction. Decision trees are easy to understand,\n\
    interpret, and visualize. They can handle both categorical and numerical features and can capture\n\
    non-linear relationships in the data."
    param_dict = {
        "criterion": {
            "help text": "function to measure the quality of a split",
            "type": list,
            "options": ["gini", "entropy"],
            "default": "gini",
            "n_search": 1
        },
        "max_depth": {
            "help text": "maximum depth of the tree",
            "type": int,
            "min": 1,
            "default": None,
            "n_search": 2
        },
        "min_samples_split": {
            "help text": "minimum number of samples required to split an internal node",
            "type": int,
            "min": 2,
            "default": 2,
            "n_search": 2
        },
        "min_samples_leaf": {
            "help text": "minimum number of samples required to be at a leaf node",
            "type": int,
            "min": 1,
            "default": 1,
            "n_search": 1
        },
        "max_features": {
            "help text": "number of features to consider when looking for the best split",
            "type": list,
            "options": ["sqrt", "log2"],
            "default": "sqrt",
            "n_search": 1
        }
    }

    def __init__(self):
        super().__init__()
        self

    

def list_available_models(): 
    return [(SVM, SVM.help_text), (RandomForest, RandomForest.help_text), (XGBoost, XGBoost.help_text), (DecisionTree, DecisionTree.help_text),
                         (NaiveBayes, NaiveBayes.help_text)]

class ModelEnsemble:
    categorical_features = ["Sex", "Embarked"]
    handle_missing_data_pipeline = {"Age": AgeHandler}
    encode_data_pipeline = Feature_Encoder
    feature_engineering_pipeline = {"Name": NameTransformer, "SimpleFeatures": SimpleFeatureGeneration, 
                                    "feature_dropper": FeatureDropper}
    metrics = [("accuracy", accuracy_score), ("recall", recall_score), ("F1Score", f1_score)]


    
    def __init__(self, model=RandomForest(),  one_hot_encoded=[], categorical_encoded=["Sex", "Embarked"]) -> None:
        super().__init__()
        self.baseModel = model
        self.classifier_model = self.baseModel.generate_model()
        missing_data_pipeline = Pipeline([(key, obj()) for key, obj in self.handle_missing_data_pipeline.items()])
        feature_engineering_pipeline = Pipeline([(key, obj()) for key, obj in self.feature_engineering_pipeline.items()])
        self.general_transform_pipeline = Pipeline([("feature engineering", feature_engineering_pipeline),
                                          ("handle missing data", missing_data_pipeline),
                                     ("encoder", self.encode_data_pipeline(one_hot_features=one_hot_encoded, categorical_features=categorical_encoded))])

    def eliminate_specific_data(self, X, y):
        embarked_index = X[X["Embarked"].isna()].index
        X.drop(index= embarked_index, inplace=True)
        y.drop(index=embarked_index, inplace=True)
        return X, y

    def generate_data(self): 
        X = pd.read_csv(os.path.join("Data", "train.csv"))
        X = X.set_index("PassengerId")
        y = X["Survived"]
        X = X.drop(columns=["Survived"])
        X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.33, random_state=42)
        X_train, y_train =self.eliminate_specific_data(X_train, y_train)
        X_test, y_test = self.eliminate_specific_data(X_test, y_test)
        return X, y, X_train, X_test, y_train, y_test

    def prepare_data(self):
        X, y, X_train, X_test, y_train, y_test = self.generate_data()
        self.general_transform_pipeline.fit(X)
        X_train = self.general_transform_pipeline.transform(X_train)
        X_test = self.general_transform_pipeline.transform(X_test)
        return X_train, X_test, y_train, y_test
    
    def calculate_metrics(self, y_test, y_pred):
        results = {}
        for metric_name, a_metric in self.metrics:
            results[metric_name]= a_metric(y_test, y_pred)
        return results

    def fit(self): 
        X_train, X_test, y_train, y_test = self.prepare_data()
        self.classifier_model.fit(X_train,y_train)
        predict = self.classifier_model.predict(X_test)
        results = self.calculate_metrics(y_test, predict)
        final_model = Pipeline([("DataTransformer", self.general_transform_pipeline),
                                ("classifier", self.classifier_model)])
        final_model.features_names = X_train.columns
        return final_model, results
    

    def grid_search(self): 
        X_train, X_test, y_train, y_test = self.prepare_data()
        grid_search_params = self.baseModel.generate_grid()
        classifier = GridSearchCV(self.classifier_model, grid_search_params, scoring='f1_macro', cv=3)
        classifier.fit(X_train, y_train)
        predict = classifier.predict(X_test)
        results = self.calculate_metrics(y_test, predict)
        final_model = Pipeline([("DataTransformer", self.general_transform_pipeline),
                                ("classifier", classifier)])
        final_model.features_names = X_train.columns
        return final_model, results


