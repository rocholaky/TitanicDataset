from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

class cliModelBase(ABC): 
    model_base= None
    classifier_name = None
    help_text = None
    param_dict = None
    @property
    def parameters(self): 
        return cliModelBase.param_dict.keys()
    
    def show_option(key): 
        help_text = cliModelBase.param_dict.get(key).get("help_text")
        type_ = cliModelBase.param_dict.get(key).get("type")
        text = f"{key} ({type_}): {help_text}\n"
        if type_ is list: 
            options = cliModelBase.param_dict.get(key).get("options")
            for i, a_option in enumerate(options): 
                text += f"{i+1}. {a_option}\n"
            text += "default"
        else: 
            min_ = cliModelBase.param_dict.get(key).get("min")
            max_ = cliModelBase.param_dict.get(key).get("max", "no max")
            default = cliModelBase.param_dict.get(key).get("default")
            text += f"min={min_}\n\
                    max={max_}\n,\
                    default={default}"
        return text
    
    def generate_model(self, parameter_kwargs):
        return self.model_base(**parameter_kwargs)
    
    




class SVM(cliModelBase): 
    model_base = SVC
    classifier_name = "SVM"
    help_text = "SVM (Support Vector Machine) is a powerful machine learning algorithm used for classification and regression tasks.\
                 It finds an optimal hyperplane to separate different classes in the data by maximizing the margin between them. \
                SVM is effective for handling high-dimensional data and can handle both linear and non-linear relationships through \
                kernel functions."
    param_dict = {"C": {"help text": "regularization term", 
                        "type": float,
                        "min": 0,
                        "default":1},
                    "kernel": {"help text": "type of kernel to use in order to apply the kernel Trick", 
                               "type": list, 
                               "options": ["linear", "poly", "rbf", "sigmoid"],
                               "default": "rbf"},
                    "degree": {"help text": "polynomial Degree of kernels (ONLY FOR POLY)", 
                               "type": int,
                               "min": 0,
                               "default": 3},
                    "gamma": {"help text": "Kernel coefficient for rbf, poly and sigmoid.",
                              "type": float,
                              "min": 0,
                              "default": "scale"}}
    
class RandomForest(cliModelBase):
    model_base = RandomForestClassifier
    classifier_name = "Random Forest"
    help_text = "Random Forest is an ensemble machine learning algorithm that uses multiple decision trees to make predictions\n.\
    It combines the predictions of individual trees to produce a final prediction. Random Forest is effective\n\
    for both classification and regression tasks. It handles high-dimensional data and can capture complex\n\
    relationships between features. It also provides measures of feature importance."
    param_dict = {"n_estimators": {"help text": "number of trees in the forest",
                                    "type": int,
                                    "min": 1,
                                    "default": 100},
            "criterion": {"help text": "function to measure the quality of a split",
            "type": list,
            "options": ["gini", "entropy"],
            "default": "gini"},
    "max_depth": {"help text": "maximum depth of the trees",
                            "type": int,
                            "min": 1,
                            "default": None},
    "min_samples_split": {"help text": "minimum number of samples required to split an internal node",
                                "type": int,
                                "min": 2,
                                "default": 2},
    "min_samples_leaf": {"help text": "minimum number of samples required to be at a leaf node",
                            "type": int,
                            "min": 1,
                            "default": 1}}
            
class LogisticRegression(cliModelBase):
        model_base = LogisticRegression
        classifier_name = "Logistic Regression"
        help_text = "Logistic Regression is a popular classification algorithm that models the relationship between the\n\
        independent variables and the dependent binary outcome using the logistic function. It is used to\n\
        predict the probability of an event occurring. Logistic Regression is suitable for binary classification\n\
        problems and can be extended to handle multi-class classification with appropriate techniques."
        param_dict = {"penalty": {"help text": "type of regularization penalty",
        "type": list,
        "options": ["l1", "l2"],
        "default": "l2"},
        "C": {"help text": "inverse of regularization strength",
        "type": float,
        "min": 0,
        "default": 1.0},
        "solver": {"help text": "algorithm to use in the optimization problem",
        "type": list,
        "options": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "default": "lbfgs"},
        "max_iter": {"help text": "maximum number of iterations",
        "type": int,
        "min": 1,
        "default": 100}}

class NaiveBayes(cliModelBase):
    model_base = GaussianNB
    classifier_name = "Naive Bayes"
    help_text = "Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with an assumption\n\
    of independence between features. It is commonly used for binary classification problems and\n\
    works well with categorical or numerical features. Naive Bayes is fast and efficient, making\n\
    it suitable for large datasets."
    param_dict = {"priors": {"help text": "prior probabilities of the classes",
                                    "type": list,
                                    "default": None},
        "var_smoothing": {"help text": "portion of the largest variance of all features added to\n\
                                        variances for calculation stability",
                            "type": float,
                            "min": 0,
                            "default": 1e-9}}
    

class XGBoost(cliModelBase):
    model_base = xgb.XGBClassifier
    classifier_name = "XGBoost"
    help_text = "XGBoost (Extreme Gradient Boosting) is a powerful gradient boosting framework widely used for\n\
    classification and regression tasks. It employs a gradient boosting algorithm that iteratively\n\
    combines weak learners (decision trees) to form a strong predictive model. XGBoost is known\n\
    for its high performance, scalability, and ability to handle complex datasets with large\n\
    numbers of features."
    param_dict = {
        "max_depth": {"help text": "maximum depth of a tree",
                                "type": int,
                                "min": 1,
                                "default": 3},
        "learning_rate": {
                        "help text": "step size shrinkage used to prevent overfitting",
                        "type": float,
                        "min": 0,
                        "default": 0.1},
        "n_estimators": {
                        "help text": "number of boosted trees to fit",
                        "type": int,
                        "min": 1,
                        "default": 100},
    "subsample": {
                "help text": "subsample ratio of the training instances",
                "type": float,
                "min": 0,
                "max": 1,
                "default": 1},
    "colsample_bytree": {
                    "help text": "subsample ratio of columns when constructing each tree",
                    "type": float,
                    "min": 0,
                    "max": 1,
                    "default": 1},
    "reg_alpha": {
                "help text": "L1 regularization term on weights",
                "type": float,
                "min": 0,
                "default": 0},
    "reg_lambda": {
                    "help text": "L2 regularization term on weights",
                    "type": float,
                    "min": 0,
                    "default": 1}}
    

class DecisionTree(cliModelBase):
    model_base = DecisionTreeClassifier
    classifier_name = "Decision Tree"
    help_text = "Decision Tree is a versatile and widely used classification algorithm that builds a tree-like model\n\
    of decisions based on features and their thresholds. It splits the data based on the feature that\n\
    provides the most information gain or Gini impurity reduction. Decision trees are easy to understand,\n\
    interpret, and visualize. They can handle both categorical and numerical features and can capture\n\
    non-linear relationships in the data."
    param_dict = {"criterion": {"help text": "function to measure the quality of a split",
                                "type": list,
                                "options": ["gini", "entropy"],
                                "default": "gini"},
                "max_depth": {"help text": "maximum depth of the tree",
                                "type": int,
                                "min": 1,
                                "default": None},
                "min_samples_split": {
                    "help text": "minimum number of samples required to split an internal node",
                    "type": int,
                    "min": 2,
                    "default": 2},
    "min_samples_leaf": {
        "help text": "minimum number of samples required to be at a leaf node",
                    "type": int,
                    "min": 1,
                    "default": 1},
    "max_features": {"help text": "number of features to consider when looking for the best split",
                    "type": list,
                    "options": ["auto", "sqrt", "log2", None],
                    "default": None}}
    



class ModelEnsambler:
    categorical_features = ["Sex", "Embarked"]
    handle_missing_data_pipeline = {"Age": AgeHandler, "Embarked": EmbarkedHandler}
    encode_data_pipeline = Feature_Encoder
    feature_engineering_pipeline = {"Name": NameTransformer, "SimpleFeatures": SimpleFeatureGeneration, 
                                    "feature_dropper": FeatureDropper}
    baseModel = RandomForest
    metrics = [("accuracy", accuracy_score), ("recall", recall_score), ("F1Score", f1_score)]

    
    def __init__(self, model_parameter_kwargs, one_hot_encoded, categorical_encoded) -> None:
        super().__init__()
        running_model = self.baseModel.generate_model(model_parameter_kwargs)
        missing_data_pipeline = Pipeline([(key, obj()) for key, obj in self.handle_missing_data_pipeline.items()])
        feature_engineering_pipeline = Pipeline([(key, obj()) for key, obj in self.feature_engineering_pipeline.items()])
        self.general_pipeline = Pipeline([("handle missing data", missing_data_pipeline),
                                     ("feature engineering", feature_engineering_pipeline),
                                     ("encoder", self.encode_data_pipeline(one_hot_features=one_hot_encoded, categorical_features=categorical_encoded)),
                                     ("classifier", running_model)])


    def generate_data(self): 
        X = pd.read_csv(os.path.join("Data", "train.csv"))
        y = X["Survived"]
        X = X.drop(columns=["survived"])
        X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def fit(self): 
        X_train, X_test, y_train, y_test = self.generate_data()
        self.general_pipeline.fit(X_train, y_train)
        predict = self.general_pipeline.predict(X_test)
        results = {}
        for metric_name, a_metric in self.metrics:
            results[metric_name]= a_metric(y_test, predict)
        return self.general_pipeline, results