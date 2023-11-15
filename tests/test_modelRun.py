import os
import sys
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
Config.parameters["Dataset"][0] = "UnitTesting"
Config.parameters["Dataloader_Variation"][0] = "Standard"
Config.parameters["MaxPerClass"][0] = 10

train, test, val = Dataload.checkAttempLoad()
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
    assert len(data_object.attacks) == Config.parameters["CLASSES"][0] - 1
    assert data_object.num_packets == len(batch[0])
    assert isinstance(data_object.unknowns, list)
    assert len(batch[0]) == len(data_object.predictions)
    assert len(batch[0]) == len(data_object.prediction_confidence)


def test_loadModelOld():
    listOfModels = helperFunctions.get_saved_models()
    Dataload.CLASSLIST = {x: Dataload.CLASSLIST[x] for x in Dataload.CLASSLIST.keys() if Dataload.CLASSLIST[x] not in ["Test", "Test2"]}
    Config.recountclasses(Dataload.CLASSLIST)
    if len(listOfModels) < 2:
        torch.manual_seed(0)
        for x in range(5):
            model.fit(1, 2, training, validation, opt_func=torch.optim.Adam)
            Config.unit_test_mode = False
            model.savePoint("Saves/models", (x + 1))
            Config.unit_test_mode = True
        listOfModels = helperFunctions.get_saved_models()
        # pytest.skip(f"Too few model savepoints to test loading. Need at least two, found {len(listOfModels)}.")
    batch = iter(training)._next_data()
    i = 1
    model.loadPoint("Saves/models/" + listOfModels[0])
    output1 = model(batch[0])
    model.loadPoint("Saves/models/" + listOfModels[i])
    output0 = model(batch[0])
    model.loadPoint("Saves/models/" + listOfModels[0])
    output2 = model(batch[0])
    for x in range(len(output1)):
        assert torch.all(output1[x] == output2[x])
    while torch.all(output0 == output1) and i + 1 < len(listOfModels):
        # Just need to show that the results are different for one version of the model.
        i += 1
        model.loadPoint("Saves/models/" + listOfModels[i])
        output0 = model(batch[0])
    assert not torch.all(output0 == output1)


def test_loadModel():
    listOfModels = helperFunctions.get_saved_models()
    Dataload.CLASSLIST = {x: Dataload.CLASSLIST[x] for x in Dataload.CLASSLIST.keys() if Dataload.CLASSLIST[x] not in ["Test", "Test2"]}
    Config.recountclasses(Dataload.CLASSLIST)
    if len(listOfModels) < 2:
        torch.manual_seed(0)
        for x in range(5):
            model.fit(1, 2, training, validation, opt_func=torch.optim.Adam)
            Config.unit_test_mode = False
            model.savePoint("Saves/models", (x + 1))
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
