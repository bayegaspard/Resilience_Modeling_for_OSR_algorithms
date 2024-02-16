import os
import sys
import datetime
# import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
test = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "src"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src/ML_Model")
import Dataload
import ModelStruct
import Config
import helperFunctions
import torch
import numpy as np
torch.manual_seed(0)
Config.set_global("dataset", "UnitTesting")
Config.set_global("dataloader_variation", "Standard")
Config.set_global("max_per_class", 10)
Config.set_global("use_renamed_packets", 0)


train, test, val = Dataload.checkAttempLoad()
torch.manual_seed(0)
training = Dataload.DataLoader(train, 100, shuffle=True, num_workers=0, pin_memory=False)
testing = Dataload.DataLoader(test, 100, shuffle=True, num_workers=0, pin_memory=False)
validation = Dataload.DataLoader(val, 100, shuffle=True, num_workers=0, pin_memory=False)
model = ModelStruct.Conv1DClassifier(mode="SoftThresh", numberOfFeatures=Dataload.getDatagroup()[0].data_length)

# This selects the default cutoff value
model.end.cutoff = 0.5
Config.unit_test_mode = True


def test_modelReads():
    batch = iter(training)
    batch = batch._next_data()
    assert isinstance(model(batch[0]), torch.Tensor)


def test_modelReads_labeledUnknowns():
    new_data = Dataload.savedPacketDataset()
    new_loader = Dataload.DataLoader(new_data, 100, shuffle=True, num_workers=0, pin_memory=False)
    batch = iter(new_loader)
    batch = batch._next_data()
    assert isinstance(model(batch[0]), torch.Tensor)


def test_modelTrain():
    output0 = model.evaluate(training)
    torch.manual_seed(0)
    model.fit(6, 0.001, training, validation, opt_func=torch.optim.Adam)
    output1 = model.evaluate(training)

    assert output0['val_loss'] > output1['val_loss']


def test_modelDataObject():
    batch = iter(training)._next_data()
    torch.manual_seed(0)
    model.fit(1, 0.001, training, validation, opt_func=torch.optim.Adam)
    data_object = model.generateDataObject(batch[0])
    assert len(data_object.attacks) == Config.get_global("CLASSES") - 1
    assert data_object.num_packets == len(batch[0])
    assert isinstance(data_object.unknowns, list)
    assert len(batch[0]) == len(data_object.predictions_numerical)
    assert len(batch[0]) == len(data_object.prediction_confidence)
    assert len(batch[0]) == len(data_object.predictions_string)


def test_loadModelOld():
    listOfModels = helperFunctions.get_saved_models()
    Dataload.LISTCLASS = {x: Dataload.LISTCLASS[x] for x in Dataload.LISTCLASS.keys() if Dataload.LISTCLASS[x] not in ["Test", "Test2"]}
    Config.recountclasses(Dataload.LISTCLASS)
    if len(listOfModels) < 2:
        torch.manual_seed(0)
        model = ModelStruct.Conv1DClassifier(mode="SoftThresh", numberOfFeatures=Dataload.getDatagroup()[0].data_length)
        torch.manual_seed(0)
        for x in range(3):
            model.fit(1, 2, training, validation, opt_func=torch.optim.Adam)
            Config.unit_test_mode = False
            model.savePoint("Saves/models", x + 1)
            Config.unit_test_mode = True
        listOfModels = helperFunctions.get_saved_models()
        # pytest.skip(f"Too few model savepoints to test loading. Need at least two, found {len(listOfModels)}.")
    else:
        torch.manual_seed(0)
        model = ModelStruct.initialize_model("Saves/models/" + listOfModels[0])

    torch.manual_seed(0)
    batch = iter(training)._next_data()
    i = 1
    Failures_to_load = 0
    Failures_to_load += model.loadPoint("Saves/models/" + listOfModels[0]) == -1
    output1 = model(batch[0])
    Failures_to_load += model.loadPoint("Saves/models/" + listOfModels[i]) == -1
    output0 = model(batch[0])
    Failures_to_load += model.loadPoint("Saves/models/" + listOfModels[0]) == -1
    output2 = model(batch[0])
    for x in range(len(output1)):
        assert torch.all(output1[x] == output2[x])
    while torch.all(output0 == output1) and i + 1 < len(listOfModels):
        # Just need to show that the results are different for one version of the model.
        i += 1
        Failures_to_load += model.loadPoint("Saves/models/" + listOfModels[i]) == -1
        output0 = model(batch[0])
    assert Failures_to_load <= len(listOfModels) - 1, "Lots of models failed for some reason"
    assert not torch.all(output0 == output1)


def test_loadModel():
    listOfModels = helperFunctions.get_saved_models()
    Dataload.LISTCLASS = {x: Dataload.LISTCLASS[x] for x in Dataload.LISTCLASS.keys() if Dataload.LISTCLASS[x] not in ["Test", "Test2"]}
    Config.recountclasses(Dataload.LISTCLASS)
    if len(listOfModels) < 2:
        torch.manual_seed(0)
        model = ModelStruct.Conv1DClassifier(mode="SoftThresh", numberOfFeatures=Dataload.getDatagroup()[0].data_length)
        torch.manual_seed(0)
        for x in range(3):
            model.fit(1, 2, training, validation, opt_func=torch.optim.Adam)
            Config.unit_test_mode = False
            model.savePoint("Saves/models", x + 1)
            Config.unit_test_mode = True
        listOfModels = helperFunctions.get_saved_models()
        # pytest.skip(f"Too few model savepoints to test loading. Need at least two, found {len(listOfModels)}.")

    torch.manual_seed(0)
    model1 = ModelStruct.Conv1DClassifier(mode="SoftThresh", numberOfFeatures=Dataload.getDatagroup()[0].data_length)
    model1.loadPoint("Saves/models/" + listOfModels[0])
    torch.manual_seed(0)
    model2 = ModelStruct.Conv1DClassifier(mode="SoftThresh", numberOfFeatures=Dataload.getDatagroup()[0].data_length)
    model2.fit(3, 2, training, validation, opt_func=torch.optim.Adam)

    # Comparison from: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/2
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return
    assert False, "Model is the same both before and after it loads"


def test_model_quicktrain():
    torch.manual_seed(0)
    mod1 = ModelStruct.Conv1DClassifier()
    torch.manual_seed(0)
    mod2 = ModelStruct.Conv1DClassifier()
    torch.manual_seed(0)
    ModelStruct.train_model(mod1)

    # Comparison from: https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/2
    for p1, p2 in zip(mod1.parameters(), mod2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return
    assert False, "Model is the same both before and after it loads"


def test_model_dates():
    listOfModels = helperFunctions.get_saved_models()
    torch.manual_seed(0)
    model = ModelStruct.initialize_model("Saves/models/" + listOfModels[0])
    assert isinstance(model.date_of_creation, datetime.datetime)


def test_model_info():
    listOfModels = helperFunctions.get_saved_models()
    for model in listOfModels:
        info = ModelStruct.get_model_info(model)

        if info.params is not None:
            assert isinstance(info.params, dict)

        if info.classes_used is not None:
            assert isinstance(info.classes_used, dict)

        if info.confusion_matrix is not None:
            assert isinstance(info.confusion_matrix, np.ndarray)

        if info.first_created is not None:
            assert isinstance(info.first_created, datetime.datetime)

        if info.last_updated is not None:
            assert isinstance(info.last_updated, datetime.datetime)

        if info.benign_f1 is not None:
            assert isinstance(info.benign_f1, float)
