import os
import sys
import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/src/ML_Model")
import Dataload
import torch
import Config
Config.parameters["Dataset"][0] = "UnitTesting"
Config.parameters["Dataloader_Variation"][0] = "Standard"


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
    Dataload.Config.unit_test_mode = True
    if len(Dataload.Config.parameters["Unknowns_clss"][0]) == 0 or Dataload.Config.parameters["LOOP"][0] != 3:   # LOOP == 3 causes problems by not having unknowns in loop level 0
        pytest.skip()
    train, test, val = Dataload.checkAttempLoad("")
    test = Dataload.DataLoader(test, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    for batch in test:
        for x in Config.parameters["Knowns_clss"][0]:
            assert x not in batch[1][1]
        for x in batch[1]:
            assert x[1] in Config.parameters["Unknowns_clss"][0]


def test_add_new_class():
    Dataload.LISTCLASS["Test"] = len(Dataload.LISTCLASS)
    Dataload.CLASSLIST[Dataload.LISTCLASS["Test"]] = "Test"
    old_number = Config.parameters["CLASSES"][0]
    Config.recountclasses(CLASSLIST=Dataload.CLASSLIST)
    new_number = Config.parameters["CLASSES"][0]
    assert old_number != new_number


def test_add_new_class_function():
    old_number_of_IDs = len(Dataload.CLASSLIST)
    old_number_of_classes = len(Dataload.LISTCLASS)
    Dataload.add_new_class("Test2")
    new_number_of_IDs = len(Dataload.CLASSLIST)
    new_number_of_classes = len(Dataload.LISTCLASS)
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
        assert val[1][0].item() in Dataload.CLASSLIST.keys()
