import unittest
from unittest.mock import patch
from unittest import mock
from TitanicML.cli import *

class CLITesting(unittest.TestCase):

    def test_listPossibleModels(self):
        with patch('builtins.input', side_effect=["2"]):
            try: 
                base_model = list_possible_models()
            except: 
                unittest.fail('The model fails to select a correct option')
        
        self.assertIsNotNone(base_model)
    
    def test_multiple_fail_listPossibleModels(self):
        # test 3 incorrect insertions and one last correct insertion
        fake_inputs = ["a", "7", " ","1"]
        with patch('builtins.input', side_effect=fake_inputs) as mock_inputs:
            base_model = list_possible_models()
        self.assertIsNotNone(base_model)


    def test_SVM_direct_training(self): 
        svm_inputs = ["1", "1", "1"]
        with patch('builtins.input', side_effect=svm_inputs) as mock_inputs:
            try: 
                cli_obj = TitanicCli("train")
            except:
                self.fail("training of svm failed")

    def test_SVM_personalized(self):
        svm_personalized_inputs = ["1", "0.1", "poly", "3", "1", "2"]
        with patch('builtins.input', side_effect=svm_personalized_inputs) as mock_inputs:
            try: 
                cli_obj = TitanicCli("train_personalized")
            except:
                self.fail("training of svm failed")

    def test_SVM_personalized_bad_inputs(self): 
        svm_personalized_inputs = ["1", "0", "0.1", "1", "rbf", "3", "0", "1", "2" ]
        with patch('builtins.input', side_effect=svm_personalized_inputs) as mock_inputs:
            try: 
                cli_obj = TitanicCli("train_personalized")
            except:
                self.fail("training of svm failed")

    def test_incorrect_titanic_action(self): 
        self.assertRaises(ValueError, TitanicCli, "train!")
        self.assertRaises(ValueError, TitanicCli, "predict")
        











if __name__ == "__main__":
    unittest.main()