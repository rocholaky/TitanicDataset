from abc import ABC
from TitanicML.model import ModelEnsemble, list_available_models


def list_possible_models(): 
        available_models = list_available_models()
        print("Here are the possible models you can choose from: \n")
        for i, (a_model, a_help_text) in enumerate(available_models): 
            print(f"{i+1}. {a_model.classifier_name}:{a_help_text}\n")
        
        while True:
            inp = input("choose the number of the model you want:")
            try: 
                inputted_value = int(inp)
                assert 0<inputted_value<len(available_models)+1, "The selected value is not a correct option number"
                base_model = available_models[inputted_value-1][0]()
            except Exception as e: 
                pass
            else: 
                break        
        return base_model

class Actions(ABC):
    action_name = None
    
    def apply_action(self, *args): 
        raise NotImplemented()

class personalizedTrain(Actions):
    action_name= "personalized_train"
    
    def __init__(self) -> None:
        super().__init__()

    def build_model(self): 
        base_model = list_possible_models()
        for a_parameter in base_model.get_parameters():
            print(f"Please Select the following parameter: {a_parameter}\n")
            verified = False
            while not verified: 
                inp = input(base_model.show_option(a_parameter))
                try: 
                    base_model.select_choice(inp)
                except:
                    pass
                else: 
                    verified = True
        return base_model
    
    def select_encoding(self): 
        one_hot_encoding = []
        categorical_encoding = []
        for a_feature in ModelEnsemble.categorical_features:
           
            verified = False
            while not verified: 
                inp = input(f"For Feature {a_feature} would you prefer what kind of encoding:\n1. One-Hot Encoding\n2. Categorical Encoding\nSelect a number:")
                try: 
                    assert inp.isnumeric()
                    assert int(inp)==2 or int(inp)==1
                    inp = int(inp)
                    if inp ==1: 
                        one_hot_encoding.append(a_feature)
                    else: 
                        categorical_encoding.append(a_feature)
                    
                except: 
                    pass
                else: 
                    verified = True
        return one_hot_encoding, categorical_encoding
                
    
    def apply_action(self, titanic_obj):
        base_model = self.build_model()
        one_hot_encoded, categorical_encoded = self.select_encoding()
        the_model = ModelEnsemble(base_model, one_hot_encoded, categorical_encoded)
        the_model, results = the_model.fit()
        titanic_obj.set_model(the_model)
        print("Training Results:")
        for key, value in results.items():
            print(f"{key}:{value}\n")
        
class Train(Actions):
    action_name = "train"

    def apply_action(self, titanic_obj):
        base_model = list_possible_models()
        the_model = ModelEnsemble()
        the_model, results = the_model.fit()
        titanic_obj.set_model(the_model)
        print("Training Results:")
        for key, value in results.items():
            print(f"{key}:{value}\n")


class TitanicCli:
    actions = {"train_personalized": personalizedTrain, "train": Train}
    model = None
    selected_action = None
    def __init__(self, action) -> None:
        super().__init__()
        self.decide_action(action)

    def decide_action(self, action_name):
        if action_name in self.actions: 
            selected_action = self.actions[action_name]()
            selected_action.apply_action(self)
        else: 
            raise ValueError("The selected Argument is not one of the provided options, you can choose between: \n"+ "\n".join([action_obj.action_name for action, action_obj in self.actions.items()]))
    def set_model(self, a_model):
        self.model = a_model