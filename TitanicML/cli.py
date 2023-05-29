from abc import ABC
from TitanicML.model import ModelEnsemble, list_available_models

class Actions(ABC):
    action_name = None
    
    def apply_action(self, *args): 
        raise NotImplemented()

class Train(Actions):
    def build_model(self): 
        available_models = list_available_models()
        print("Here are the possible models you can choose from: \n")
        for i, a_model in enumerate(available_models): 
            print(f"{i+1}. {a_model.classifier_name}\n")
        
        while True:
            inp = input("choose the number of the model you want:")
            try: 
                inputted_value = int(inp)
                assert 0<inputted_value<len(available_models)+1, "The selected value is not a correct option number"
                base_model = available_models[inputted_value]
            except: 
                pass
            else: 
                break
        for a_parameter in base_model.parameters:
            verified = False
            while not verified: 
                inp = input(base_model.show_options(a_parameter))
                try: 
                    base_model.select_choice(inp)
                except:
                    pass
                else: 
                    verified = True
        return base_model
    
    def select_encoding(): 
        one_hot_encoding = []
        categorical_encoding = []
        for a_feature in ModelEnsemble.categorical_features:
            inp = input(f"For Feature {a_feature} would you prefer what kind of encoding:\n\
                        1. One-Hot Encoding\
                        2. Categorical Encoding\n\
                        Select a number:")
            verified = False
            while not verified: 
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
                
    
    def apply_action(self, TitanicCli_obj):
        base_model = self.build_model()
        one_hot_encoded, categorical_encoded = self.select_encoding()
        the_model = ModelEnsemble(base_model, one_hot_encoded, categorical_encoded)
        the_model.fit()
        TitanicCli.set_model(the_model)

        
    

class Save(Actions):
    def apply_action(self, TitanicCli_obj, path):
        pass
        
    
class Predict(Actions):
    def apply_action(self, *args):
        return super().apply_action(*args)


class TitanicCli:
    actions = {"train": Train, "save": Save, "predict": Predict}
    model = None
    selected_action = None
    def __init__(self, action) -> None:
        super().__init__()
        self.decide_action(action)

    def decide_action(self, action_name):
        if action_name in self.actions: 
            selected_action = self.actions[action_name]
            selected_action.apply_action()
        else: 
            raise ValueError("The selected Argument is not one of the provided options, you can choose between: \n"+ "\n".join([action.action_name for action in self.actions]))
    def set_model(self, a_model):
        self.model = a_model