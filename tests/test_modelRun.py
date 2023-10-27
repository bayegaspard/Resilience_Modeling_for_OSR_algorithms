import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
test = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "src"))
import Dataload
import ModelStruct
import Config
import torch


train, test, val = Dataload.checkAttempLoad()
training = Dataload.DataLoader(train, 100, shuffle=True, num_workers=0, pin_memory=False)
testing = Dataload.DataLoader(test, 100, shuffle=True, num_workers=0, pin_memory=False)
validation = Dataload.DataLoader(val, 100, shuffle=True, num_workers=0, pin_memory=False)
model = ModelStruct.Conv1DClassifier(mode="SoftThresh", numberOfFeatures=Dataload.getDatagroup()[0].data_length)

# This selects the default cutoff value
model.end.cutoff = 0.5


def test_modelReads():
    batch = iter(training)
    batch = batch._next_data()
    assert isinstance(model(batch[0]), torch.Tensor)


def test_modelDataObject():
    batch = iter(training)._next_data()
    model.fit(1, 0.001, training, validation, opt_func=torch.optim.Adam)
    data_object = model.generateDataObject(batch[0])
    assert len(data_object.attacks) == Config.parameters["CLASSES"][0] - 1
    assert data_object.num_packets == len(batch[0])
    assert isinstance(data_object.unknowns, list)
