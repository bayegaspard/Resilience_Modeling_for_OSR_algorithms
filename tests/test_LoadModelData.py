import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src")
import Dataload
import torch
import Config


def test_checkAttempLoad():
    train, test, val = Dataload.checkAttempLoad()
    assert isinstance(train, Dataload.Dataset)
    assert isinstance(test, Dataload.Dataset)
    assert isinstance(val, Dataload.Dataset)


def test_createLoaders():
    train, test, val = Dataload.checkAttempLoad()
    training = Dataload.DataLoader(train, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    testing = Dataload.DataLoader(test, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    validation = Dataload.DataLoader(val, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    dataloaders = [training, testing, validation]
    for data in dataloaders:
        for batch in data:
            assert isinstance(batch[0], torch.Tensor)
            assert isinstance(batch[1], torch.Tensor)
            # test = batch[1]
            assert batch[1].shape[1] == 2


def test_validation_dataset():
    train, test, val = Dataload.checkAttempLoad("")
    val = Dataload.DataLoader(val, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    for batch in val:
        for x in Config.parameters["Unknowns_clss"][0]:
            assert x not in batch[1][1]
        for x in batch[1]:
            assert x[1] in Config.parameters["Knowns_clss"][0]


def test_testing_dataset():
    Dataload.Config.parameters["Mix unknowns and validation"][0] = 0
    assert Dataload.Config.parameters["Mix unknowns and validation"][0] != 3  # LOOP == 3 causes problems by not having unknowns in loop level 0
    Dataload.Config.unit_test_mode = True
    train, test, val = Dataload.checkAttempLoad("")
    test = Dataload.DataLoader(test, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    for batch in test:
        for x in Config.parameters["Knowns_clss"][0]:
            assert x not in batch[1][1]
        for x in batch[1]:
            assert x[1] in Config.parameters["Unknowns_clss"][0]
