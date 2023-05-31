import unittest
from TitanicML.model import *


class ModelEnsembleTest(unittest.TestCase):

    def test_model_ensemble(self):
        model = ModelEnsemble()
        classifier_model, results = model.fit()
        
        # Verify that the classifier model is not None
        self.assertIsNotNone(classifier_model)

        # Verify that the results dictionary contains the expected metrics
        self.assertIn("accuracy", results)
        self.assertIn("recall", results)
        self.assertIn("F1Score", results)

        # Verify that the accuracy, recall, and F1 scores are within the expected range
        self.assertGreaterEqual(results["accuracy"], 0)
        self.assertLessEqual(results["accuracy"], 1)

        self.assertGreaterEqual(results["recall"], 0)
        self.assertLessEqual(results["recall"], 1)

        self.assertGreaterEqual(results["F1Score"], 0)
        self.assertLessEqual(results["F1Score"], 1)



    def test_list_available_models(self):
        available_models = list_available_models()
        
        # Verify that the list is not empty
        self.assertGreater(len(available_models), 0)

        # Verify that each model in the list has a classifier name and help text
        for model, help_text in available_models:
            self.assertIsNotNone(model.classifier_name)
            self.assertIsNotNone(help_text)
            self.assertIsNotNone(model.param_dict)

    ## Test default training: 
    def test_default_training(self): 
        available_models = list_available_models()
        # Verify that each model in the list has a classifier name and help text
        for model, _ in available_models:
            model = model()
            m = ModelEnsemble(model)
            classifier_model, results = m.fit()
            
            # Verify that the classifier model is not None
            self.assertIsNotNone(classifier_model)

            # Verify that the results dictionary contains the expected metrics
            self.assertIn("accuracy", results)
            self.assertIn("recall", results)
            self.assertIn("F1Score", results)

    ## Test default training: 
    def test_grid_search(self): 
        # testing just one model: 
        available_models = [(XGBoost, XGBoost.help_text)]
        # Verify that each model in the list has a classifier name and help text
        for model, _ in available_models:
            model = model()
            m = ModelEnsemble(model)
            classifier_model, results = m.grid_search()
            
            # Verify that the classifier model is not None
            self.assertIsNotNone(classifier_model)

            # Verify that the results dictionary contains the expected metrics
            self.assertIn("accuracy", results)
            self.assertIn("recall", results)
            self.assertIn("F1Score", results)

    ## Testing personalized training: 
    def test_personalized_SVM(self):
        model = SVM()
        model.selected_parameters = {
            "C": 1.0,
            "kernel": "rbf",
            "degree": 3
        }
        build_classifier = model.generate_model()
        self.assertIsNotNone(build_classifier)
        self.assertEqual(build_classifier.C, 1.0)
        self.assertEqual(build_classifier.kernel, "rbf")
        self.assertEqual(build_classifier.degree, 3)
        m = ModelEnsemble(model)
        classifier_model, results = m.fit()
        
        # Verify that the classifier model is not None
        self.assertIsNotNone(classifier_model)

        # Verify that the results dictionary contains the expected metrics
        self.assertIn("accuracy", results)
        self.assertIn("recall", results)
        self.assertIn("F1Score", results)

    def test_personalized_NaiveBayes(self):
        model = NaiveBayes()
        model.selected_parameters = {
            "var_smoothing": 1e-9
        }
        build_classifier = model.generate_model()
        self.assertIsNotNone(build_classifier)
        self.assertEqual(build_classifier.var_smoothing, 1e-9)
        m = ModelEnsemble(model)
        classifier_model, results = m.fit()
        
        # Verify that the classifier model is not None
        self.assertIsNotNone(classifier_model)

        # Verify that the results dictionary contains the expected metrics
        self.assertIn("accuracy", results)
        self.assertIn("recall", results)
        self.assertIn("F1Score", results)

    
    def test_personalized_RandomForest(self):
        model = RandomForest()
        model.selected_parameters = {
            "n_estimators": 20,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        }
        build_classifier = model.generate_model()
        self.assertIsNotNone(build_classifier)
        self.assertEqual(build_classifier.n_estimators, 20)
        self.assertEqual(build_classifier.criterion, "gini")
        self.assertIsNone(build_classifier.max_depth)
        self.assertEqual(build_classifier.min_samples_split, 2)
        self.assertEqual(build_classifier.min_samples_leaf, 1)
        m = ModelEnsemble(model)
        classifier_model, results = m.fit()
        
        # Verify that the classifier model is not None
        self.assertIsNotNone(classifier_model)

        # Verify that the results dictionary contains the expected metrics
        self.assertIn("accuracy", results)
        self.assertIn("recall", results)
        self.assertIn("F1Score", results)

    def test_personalized_XGBoost(self):
        model = XGBoost()
        model.selected_parameters = {
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0,
            "reg_lambda": 1
        }
        build_classifier = model.generate_model()
        self.assertIsNotNone(build_classifier)
        self.assertEqual(build_classifier.max_depth, 3)
        self.assertEqual(build_classifier.learning_rate, 0.1)
        self.assertEqual(build_classifier.n_estimators, 100)
        self.assertEqual(build_classifier.subsample, 1.0)
        self.assertEqual(build_classifier.colsample_bytree, 1.0)
        self.assertEqual(build_classifier.reg_alpha, 0)
        self.assertEqual(build_classifier.reg_lambda, 1)
        m = ModelEnsemble(model)
        classifier_model, results = m.fit()
        
        # Verify that the classifier model is not None
        self.assertIsNotNone(classifier_model)

        # Verify that the results dictionary contains the expected metrics
        self.assertIn("accuracy", results)
        self.assertIn("recall", results)
        self.assertIn("F1Score", results)

    def test_personalized_DecisionTree(self):
        model = DecisionTree()
        model.selected_parameters = {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt"
        }
        build_classifier = model.generate_model()
        self.assertIsNotNone(build_classifier)
        self.assertEqual(build_classifier.criterion, "gini")
        self.assertIsNone(build_classifier.max_depth)
        self.assertEqual(build_classifier.min_samples_split, 2)
        self.assertEqual(build_classifier.min_samples_leaf, 1)
        self.assertEqual(build_classifier.max_features, "sqrt")
        m = ModelEnsemble(model)
        classifier_model, results = m.fit()
        
        # Verify that the classifier model is not None
        self.assertIsNotNone(classifier_model)

        # Verify that the results dictionary contains the expected metrics
        self.assertIn("accuracy", results)
        self.assertIn("recall", results)
        self.assertIn("F1Score", results)



class ClassifierModelTest(unittest.TestCase): 

    # test
    def test_classification_models_attributes(self):
        # get available classification models
        available_models = list_available_models()
        # iterate through them: 
        for model, help_text in available_models:
            # instantiate model
            model_instance = model()
            # check that the model selection dict is not filled
            self.assertEqual(len(model_instance.selected_parameters), 0)
            # check that the model can be generated without initial values
            self.assertIsNotNone(model_instance.generate_model())
            # check that the model help_text is not empty
            self.assertNotEqual(help_text, "")
            # check that all models have parameters:
            self.assertNotEqual(len(list(model_instance.get_parameters())), 0)
            # check that all options are showable:
            for a_param in model_instance.get_parameters():
                with self.subTest(value=a_param, msg=f"Testing parameter: {a_param}"):
                    self.assertIsInstance(model_instance.show_option(a_param), str)

    
    def test_select_choice(self):
        model_instance = RandomForest()
        for a_param in model_instance.get_parameters():
            choices = model_instance.param_dict[a_param]
            default = choices["default"]
            with self.subTest(value=a_param, msg=f"Testing parameter: {a_param} with default: {default}"):
                try: 
                    model_instance.select_choice(default)
                except: 
                    self.fail("the default value returned error")




if __name__ == "__main__":
    unittest.main()
