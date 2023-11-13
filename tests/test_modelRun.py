import os
import sys
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
test = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "src"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src/ML_Model")
import Dataload
import ModelStruct
import Config
import helperFunctions
import torch


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
    model.fit(10, 0.001, training, validation, opt_func=torch.optim.Adam)
    output1 = model.evaluate(training)

    assert output0['val_loss'] > output1['val_loss']


def test_modelDataObject():
    batch = iter(training)._next_data()
    model.fit(1, 0.001, training, validation, opt_func=torch.optim.Adam)
    data_object = model.generateDataObject(batch[0])
    assert len(data_object.attacks) == Config.parameters["CLASSES"][0] - 1
    assert data_object.num_packets == len(batch[0])
    assert isinstance(data_object.unknowns, list)


def test_loadModel():
    listOfModels = helperFunctions.get_saved_models()
    if len(listOfModels) < 2:
        pytest.skip(f"Too few model savepoints to test loading. Need at least two, found {len(listOfModels)}.")
    batch = iter(training)._next_data()
    model.loadPoint("Saves/models/" + listOfModels[2])
    output0 = model(batch[0])
    model.loadPoint("Saves/models/" + listOfModels[0])
    output1 = model(batch[0])
    model.loadPoint("Saves/models/" + listOfModels[0])
    output2 = model(batch[0])
    for x in range(len(output1)):
        assert torch.all(output1[x] == output2[x])
        assert not torch.all(output0[x] == output1[x])
