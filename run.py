import argparse
from TitanicML.cli import TitanicCli




if __name__=="__main__":
    available_commands = ["start"]
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, help="command to use in the package")
    #parser.add_argument("--device", type=str, help="cuda device in which we want to run the model", default='cpu')
    args = parser.parse_args()


