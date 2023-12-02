import sys
sys.path.append("src/ML_Model")
import Config
import Dataload
import Distance_Types
import helperFunctions
import EndLayer
import ModelStruct

if False:
    Dataload
    Distance_Types
    helperFunctions
    EndLayer

if __name__ == "__main__":
    model_list = {"Convolutional": ModelStruct.Conv1DClassifier, "Fully_Connected": ModelStruct.FullyConnected}
    model = model_list[Config.parameters["model"][0]](mode=Config.parameters["OOD Type"][0])
    assert isinstance(model, ModelStruct.AttackTrainingClassification)
    ModelStruct.train_model(model)
    model.savePoint("Saves/models/" + Config.parameters["Saveloc"][0], exact_name=True)
