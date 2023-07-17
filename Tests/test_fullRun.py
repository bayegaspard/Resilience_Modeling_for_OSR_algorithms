#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sys.path.append(f"{root_folder}/src/main")
sys.path.append(f"{root_folder}/src/CodeFromImplementations")
import src.main.EndLayer as EndLayer
# import src.main.Config as Config
import src.main.main as main
import torch
import numpy as np
import src.main.FileHandling as FileHandling
import src.main.Dataload as Dataload
import src.main.ModelStruct as ModelStruct
from torch.utils.data import DataLoader

main.Config.parameters["num_epochs"][0] = 1
main.Config.parameters["MaxPerClass"][0] = 10
main.Config.parameters["LOOP"][0] = 0

def testrun():
    """
    Tests if everything runs properly in a single run.
    """
    main.Config.unit_test_mode = True
    main.run_model(graphDefault=False)

def testrunall():
    """
    Tries all of the algorithms. Except Openmax
    """
    main.Config.unit_test_mode = True
    if "Open" in main.Config.alg:
        main.Config.alg.remove("Open")
    for x in main.Config.alg:
        main.Config.parameters["OOD Type"][0] = x
        main.run_model(graphDefault=False)

def testrunEnergy():
    """
    Tries Energy as it gets some of the more intresting outputs even if it does not train much.
    """
    main.Config.unit_test_mode = True
    main.Config.parameters["OOD Type"][0] = "Energy"
    main.run_model(graphDefault=False)

def testrunDOC():
    """
    DOC is being problematic.
    """
    main.Config.unit_test_mode = True
    main.Config.parameters["OOD Type"][0] = "DOC"
    main.run_model(graphDefault=False)

def testrunOpen():
    """
    Openmax is being problematic.
    """
    main.Config.unit_test_mode = True
    main.Config.parameters["OOD Type"][0] = "Open"
    main.run_model(graphDefault=False)

def testLoadDataset():
    main.torch.manual_seed(1)
    train1, test1, val1 = FileHandling.checkAttempLoad("")
    main.torch.manual_seed(1)
    train2, test2, val2 = FileHandling.checkAttempLoad("")
    for x,y in zip(train1,train2):
        assert torch.all(x[0] == y[0])
    
def testLoadDatasetfromSave():
    main.Config.parameters["attemptLoad"][0] = 1
    main.torch.manual_seed(1)
    train1, test1, val1 = FileHandling.checkAttempLoad("")
    main.torch.manual_seed(1)
    train2, test2, val2 = FileHandling.checkAttempLoad("")
    for x,y in zip(train1,train2):
        assert torch.all(x[0] == y[0])

def testfindSave():
    """Finds if the save exists"""
    epochFound = ModelStruct.AttackTrainingClassification.findloadEpoch()
    assert epochFound != -1 #CANNOT FIND MODEL
        
    pathFound = ModelStruct.AttackTrainingClassification.findloadPath(epochFound)
    assert os.path.exists(pathFound) #The path is invalid.

def testrunFromSave():
    """
    Tests if saves work and if they result in the same answer if given the same seed.
    """
    global vals
    vals = {}
    def addtoLoopNames(itemDescription,item):
        global vals
        assert isinstance(itemDescription,str)
        vals[itemDescription] = item
    def checkifinloop(itemDescription,item):
        global vals
        assert item==(vals[itemDescription]) or (item is np.nan and vals[itemDescription] is np.nan)
    main.Config.unit_test_mode = True
    main.Config.parameters["num_epochs"][0] = 0
    main.torch.manual_seed(1)
    main.run_model(addtoLoopNames,graphDefault=False)
    main.torch.manual_seed(1)
    main.run_model(checkifinloop,graphDefault=False)
