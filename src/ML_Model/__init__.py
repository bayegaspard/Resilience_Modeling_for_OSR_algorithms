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
    model = model_list[Config.get_global("model")](mode=Config.get_global("OOD_type"))
    assert isinstance(model, ModelStruct.AttackTrainingClassification)
    ModelStruct.train_model(model)
    model.savePoint("Saves/models/" + Config.get_global("save_loc"), exact_name=True)
