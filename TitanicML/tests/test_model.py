import unittest
from sklearn.metrics import accuracy_score, recall_score, f1_score
from TitanicML.model import ModelEnsemble, list_available_models


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
            self.assertEquals(len(model.selected_parameters), 0)
            self.assertIsNotNone(model.generate_model())
            self.assertNotEquals(len(list(model.get_parameters()), 0))

if __name__ == "__main__":
    unittest.main()
