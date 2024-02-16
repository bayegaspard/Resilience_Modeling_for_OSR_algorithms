import os
import sys
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src/ML_Model")
import Dataload
import torch
import Config
Config.parameters["dataset"][0] = "UnitTesting"
Config.parameters["dataloader_variation"][0] = "Standard"


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
        for x in Config.parameters["unknowns_clss"][0]:
            assert x not in batch[1][1]
        for x in batch[1]:
            assert x[1] in Config.parameters["knowns_clss"][0]


def test_testing_dataset():
    Dataload.Config.parameters["mix_unknowns_and_validation"][0] = 0
    Dataload.Config.unit_test_mode = True
    if len(Dataload.Config.parameters["unknowns_clss"][0]) == 0 or Dataload.Config.parameters["LOOP"][0] != 3:   # LOOP == 3 causes problems by not having unknowns in loop level 0
        pytest.skip()
    train, test, val = Dataload.checkAttempLoad("")
    test = Dataload.DataLoader(test, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    for batch in test:
        for x in Config.parameters["knowns_clss"][0]:
            assert x not in batch[1][1]
        for x in batch[1]:
            assert x[1] in Config.parameters["unknowns_clss"][0]


def test_add_new_class():
    Dataload.CLASSLIST["Test"] = len(Dataload.CLASSLIST)
    Dataload.LISTCLASS[Dataload.CLASSLIST["Test"]] = "Test"
    old_number = Config.parameters["CLASSES"][0]
    Config.recountclasses(CLASSLIST=Dataload.LISTCLASS)
    new_number = Config.parameters["CLASSES"][0]
    assert old_number != new_number


def test_add_new_class_function():
    old_number_of_IDs = len(Dataload.LISTCLASS)
    old_number_of_classes = len(Dataload.CLASSLIST)
    Dataload.add_new_class("Test2")
    new_number_of_IDs = len(Dataload.LISTCLASS)
    new_number_of_classes = len(Dataload.CLASSLIST)
    assert old_number_of_IDs != new_number_of_IDs
    assert old_number_of_classes != new_number_of_classes


def test_dynamic_dataloader():
    # old_number = Config.parameters["CLASSES"][0]
    data = Dataload.savedPacketDataset()
    # new_number = Config.parameters["CLASSES"][0]
    dl = Dataload.DataLoader(data, 100, shuffle=True, num_workers=0, pin_memory=False)
    batch = iter(dl)._next_data()
    for val in zip(batch[0], batch[1]):
        assert val[1][0] == val[1][1]
        assert val[1][0] < Config.parameters["CLASSES"][0]
        assert len(val[0]) == 1504
        assert val[1][0].item() in Dataload.LISTCLASS.keys()
