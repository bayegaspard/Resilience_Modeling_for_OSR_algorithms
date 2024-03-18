import torch
import sys
import argparse
import numpy as np


if __name__ != "Config":
    # if "Config" in sys.modules:
    #     class doubleImport(ImportError):
    #         """
    #         Config was imported using a different path after it has already been imported.
    #         This causes problems when Config is modified.
    #         """
    #         pass
    #     raise doubleImport
    print(f"A POSSIBLE PROBLEM HAS OCCURED, Config was loaded improperly, from {__name__} instead of directly\
    this might break some global variables by having two copies", file=sys.stderr)

# TODO: Rework config so that it is less janky and uses less bad practices of global variables.
# Possibly by moving HelperFunctions Loop functions to outside of the program
# and just using the command line parser for the individual sections.

# This config file is mainly used as global variables for the rest of the program.
# It should only be modified by the loop commands in helperfunctions


def loopOverUnknowns(unknownlist=False):
    """
    Given a list of unknowns (integers 0-14) this will create a list of knowns (the inverted list).
    """
    if unknownlist is False:
        unknownlist = parameters["unknowns_clss"][0]
    knownVals = list(range(parameters["CLASSES"][0]))
    notused = unknownlist + UnusedClasses
    notused.sort()
    for un in notused:
        if un in knownVals:
            knownVals.remove(un)

    if len(unknownlist) > parameters["CLASSES"][0] - 3:
        print("Too many unknowns, some algorithms might not work")
    if len(knownVals) < 2:
        print("Too few knowns, things might break")
    parameters["unknowns_clss"] = [unknownlist, "Values used for testing"]
    parameters["knowns_clss"] = [knownVals, "Values used for training"]
    return knownVals


def recountclasses(CLASSLIST: dict):
    parameters["CLASSES"][0] = len(CLASSLIST)
    loopOverUnknowns()


# This is the diffrent optimization functions
opt_func = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD, "RMSprop": torch.optim.RMSprop}

parameter_name_conversions = {
    "Mix unknowns and validation": "mix_unknowns_and_validation",
    "attemptLoadModel": "attempt_load_model",
    "attemptLoadData": "attempt_load_data",
    "testlength": "test_length",
    "MaxPerClass": "max_per_class",
    "learningRate": "learning_rate",
    "OOD Type": "OOD_type",
    "Dropout": "dropout",
    "Dataloader_Variation": "dataloader_variation",
    "Unknowns_clss": "unknowns_clss",
    "temperature": "temperature",
    "Degree of Overcompleteness": "degree_of_overcompleteness",
    "Number of Layers": "number_of_layers",
    "Nodes": "nodes",
    "Activation": "activation",
    "Dataset": "dataset",
    "SchedulerStepSize": "scheduler_num_epochs",
    "SchedulerStep": "scheduler_step_distance",
    "Var_filtering_threshold": "var_filtering_threshold",
    "Experimental_bitConvolution": "experimental_bit_convolution",
    "Saveloc": "save_loc",
    "Knowns_clss": "knowns_clss"
}

# Here are all of the paremeters for the model.
parameters = {
    # These parameters are orginized like this:
    # "ParamName": [Value, "Description", [possible values]]
    # for a parameter called "ParamName" with a value of Value
    "batch_size": [100, "Number of items per batch", [], int],
    "num_workers": [0, "Number of threads working on building batches", [], int],
    "attempt_load_model": [0, "0: do not use saves for the model\n1:use saves for the model", [0, 1], int],
    "attempt_load_data": [0, "0: do not use saves for the dataset\n1:use saves for the dataset", [0, 1], int],
    "test_length": [1 / 4, "[0, 1) percentage of training to test with", [], float],
    "mix_unknowns_and_validation": [1, "0 or 1, 0 means that the test set is purely unknowns and 1 means that the testset is the validation set plus unknowns (for testing)", [0, 1], int],
    "max_per_class": [100, "Maximum number of samples per class\n if Dataloader_Variation is Cluster and this value is a float it interprets it as the maximum percentage of the class instead.", [], int],
    "num_epochs": [50, "Number of times it trains on the whole trainset", [], int],
    "learning_rate": [0.01, "a modifier for training", [], float],
    "threshold": [0.5, "When to declare something to be unknown", [], float],
    "model": ["Convolutional", "Model type", ["Fully_Connected", "Convolutional"], str],
    "OOD_type": ["Soft", "type of out of distribution detection", ["Soft", "Open", "Energy", "COOL", "DOC", "iiMod"], str],
    "dropout": [0.1, "percent of nodes that are skipped per run, larger numbers for more complex models [0, 1)", [], float],
    "dataloader_variation": ["Standard", "Defines the style of Dataloader used. This affects sampling from the dataset", ["Standard", "Cluster", "Flows"], str],
    "optimizer": ["Adam", "Defines which optimizer to use.", ["Adam", "SGD", "RMSprop"], str],
    "unknowns_clss": [[], "Class indexes used as unknowns.", [], list],
    "CLASSES": [15, "Number of classes, do not change", [10, 15], int],
    "temperature": [1, "Energy OOD scaling parameter", [], float],
    "degree_of_overcompleteness": [3, "Parameter for Fitted Learning", [], int],
    "number_of_layers": [3, "Number of layers to add to the base model", [], int],
    "nodes": [512, "The number of nodes per added layer", [], int],
    "activation": ["Leaky", "The type of activation function to use", ["ReLU", "Tanh", "Sigmoid", "Leaky"], str],
    "LOOP": [0, "This is a parameter that determines if we want to loop over the algorithms.\n "\
                "0: no loop, 1: loop through variations of algorithms, thresholds, learning rates, groups and numbers of epochs, \n"\
                "2: Loop while adding more unknowns into the training data (making them knowns) without resetting the model, \n"\
                "3: Loop through different data distributions without training the model.\n"\
                "4: Loop through predefined hyperparameters found in datasets/hyperparamList.csv", [0, 1, 2, 3, 4], int],
    "dataset": ["Payload_data_CICIDS2017", "This is what dataset we are using, ", ["Payload_data_CICIDS2017", "Payload_data_UNSW"], str],
    "scheduler_num_epochs": [10, "This is how often the scheduler takes a step, 3 means every third epoch", [], int],
    "scheduler_step_distance": [0.9, "This is how big a step the scheduler takes, leave 0 for no step", [], float],
    "var_filtering_threshold": [-20, "If not -1, the model will first apply a varmax layer to the endlayer to know if the 'OOD_type' algorithm should be applied. This will use the number given as the threshold.", [], str],
    "experimental_bit_convolution": [0, "Convolutional network contains an experimental 2 bit convolution over the bits", [0, 1], int],
    "save_loc": ["defaultSave.pth", "Save location for running from __init__", [], str],
    "use_renamed_packets": [0, "When creating the model, use the renamed packets for clases", [0, 1], int],
    "Padec": [1, "Toggle whether or not you want to pass data to PADEC", [0, 1], int],
    "Padec_Values": [[4,3], "List what classes you want the PADEC model to describe", [], list]
}


# Argparse tutorial: https: //docs.python.org/3/howto/argparse.html
parser = argparse.ArgumentParser()
for x in parameters.keys():
    if len(parameters[x][2]) == 0:
        parser.add_argument(f"--{x}", type=parameters[x][3], default=parameters[x][0], help=parameters[x][1], required=False)
    else:
        parser.add_argument(f"--{x}", type=parameters[x][3], choices=parameters[x][2], default=parameters[x][0], help=parameters[x][1], required=False)

if "pytest" not in sys.modules:  # The argument parser appears to have issues with the pytest tests. I have no idea why.
    args = parser.parse_args()
    for x in args._get_kwargs():
        parameters[x[0]][0] = x[1]

if isinstance(parameters["unknowns_clss"][0], str):
    if len(parameters["unknowns_clss"][0]) > 0 and len(parameters["unknowns_clss"][0]) != 2:  # Not sure why I need this specifier but it breaks if the default is []
        # print(len(parameters["unknowns_clss"][0]))
        parameters["unknowns_clss"][0] = [int(y) for y in parameters["unknowns_clss"][0].removesuffix("]").removeprefix("[").split(sep=", ")]
    else:
        parameters["unknowns_clss"][0] = []

if isinstance(parameters["var_filtering_threshold"][0], str):
    if len(parameters["var_filtering_threshold"][0]) > 0 and ("," in parameters["var_filtering_threshold"][0]):
        if len(parameters["var_filtering_threshold"][0]) != 2:  # Not sure why I need this specifier but it breaks if the default is []
            # print(len(parameters["unknowns_clss"][0]))
            parameters["var_filtering_threshold"][0] = [float(y) for y in parameters["var_filtering_threshold"][0].removesuffix("]").removeprefix("[").split(sep=", ")]
    elif len(parameters["var_filtering_threshold"][0]) > 0 and ("," not in parameters["var_filtering_threshold"][0]):
        parameters["var_filtering_threshold"][0] = float(parameters["var_filtering_threshold"][0])
    else:
        parameters["var_filtering_threshold"][0] = [-1, -1]


DOC_kernels = [3, 4, 5]

# Set Number of classes:
if parameters["dataset"][0] == "Payload_data_UNSW":
    parameters["CLASSES"][0] = 10
    UnusedClasses = []
else:
    UnusedClasses = [8, 9, 10]
UnusedClasses = []
parameters["optimizer"][0] = opt_func[parameters["optimizer"][0]]

# Dendrogram chunk uses a slightly diffrent output on the model structure.
# (Also, dendrogram chunk is not working, so don't use it. Possibly related.)
if parameters["dataloader_variation"][0] == "Old_Cluster":
    parameters["CLASSES"][0] = parameters["CLASSES"][0] * 32


# Add a value to the dictionary that is the inverse of the unknowns
loopOverUnknowns()


# This is for saving the original number of epochs
num_epochs = parameters["num_epochs"][0]


# This is to test all of the algorithms one after the other. (Loop 1 values)
alg = ["Soft", "Open", "Energy", "COOL", "DOC", "iiMod"]
batch = [100, 1000, 10000, 100000]
datapoints_per_class = [10, 100, 1000]
thresholds = [0.1, 1, 10]
thresholds = [30, 20, 15, 5]
thresholds = [parameters["threshold"][0]]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
activation = ["ReLU", "Tanh", "Sigmoid", "Leaky"]
groups = [[], [2], [2, 3], [2, 3, 4], [2, 3, 4, 5], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]]
# groups = [[7, 8, 9]]
if parameters["dataset"][0] == "Payload_data_CICIDS2017":
    incGroups = [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [6, 7, 8, 9, 10, 11, 12, 13, 14], [7, 8, 9, 10, 11, 12, 13, 14], [8, 9, 10, 11, 12, 13, 14], [9, 10, 11, 12, 13, 14], [10, 11, 12, 13, 14], [11, 12, 13, 14], [12, 13, 14], [13, 14], [14]]
# This one list is for loop 2. Note: array size should be decreasing.
else:
    incGroups = [[2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9], [5, 6, 7, 8, 9], [6, 7, 8, 9], [7, 8, 9], [8, 9], [9]]
epochs = []
epochs = [1, 10, 100, 150]


# groups = [list(range(2, parameters["CLASSES"][0]))]
# # Little bit of code that generates incremental numbers of unknowns.
# while len(groups[0])>2:
#     new = groups[0].copy()
#     new.pop(0)
#     new.pop(0)
#     groups.insert(0, new)
# # Little bit of code that generates decrementing numbers of unknowns.
# incGroups = [list(range(2, parameters["CLASSES"][0]))]
# while len(incGroups[-1])>1:
#     new = incGroups[-1].copy()
#     new.pop(0)
#     incGroups.append(new)

# Here is where we remove some of the algorithms if we want to skip them. We could also just remove them from the list above.
# alg.remove("Soft")
# alg.remove("Open")
# alg.remove("Energy")
# alg.remove("COOL")
# alg.remove("DOC")
# alg.remove("iiMod")

# it causes problems if you dont start at the start of the loop.
if parameters["LOOP"][0] == 1:
    parameters["OOD_type"][0] = alg[0]


# Optimizer has been removed from the list of things we are changing
optim = [opt_func["Adam"], opt_func["SGD"], opt_func["RMSprop"]]

# This is an array to eaiser loop through everything.
loops = [batch, learning_rates, activation, ["Standard", "Cluster"], groups]
# loops = [groups]
loops2 = ["batch_size", "learning_rate", "activation", "dataloader_variation", "Unknowns"]
# loops2 = ["Unknowns"]
for i in range(len(loops)):
    if loops2[i] == "Unknowns":
        loops[i].insert(0, parameters["unknowns_clss"][0])
    elif loops2[i] == "optimizer":
        loops[i].insert(0, parameters[loops2[i]])
    elif loops2[i] == "None":
        pass
    else:
        loops[i].insert(0, parameters[loops2[i]][0])

# Override the unknowns because model is kept
if parameters["LOOP"][0] == 2:
    parameters["unknowns_clss"][0] = incGroups[0]
    parameters["Unknowns"] = f"{incGroups[0]} Unknowns"
    parameters["knowns_clss"][0] = loopOverUnknowns()


# This controls all of the save data (so that we can run small tests without changing the nice files.)
unit_test_mode = False

use_alg_thesholds = False


if parameters["LOOP"][0] == 3:
    print("Warning: Unknowns may have been changed due to LOOP 3 percentages file")
    import pandas as pd
    # parameters["num_epochs"][0] = 0
    parameters["loopLevel"] = [0, "What percentages the model is on"]
    parameters["MaxSamples"] = [parameters["max_per_class"][0], "Max number of samples total"]
    file = pd.read_csv("datasets/percentages.csv", index_col=None).to_numpy()
    zeros = file[0][:parameters["CLASSES"][0]] == 0
    unknownClasses = zeros.nonzero()[0]
    parameters["unknowns_clss"][0] = unknownClasses.tolist()


# Getting version number
# https: //gist.github.com/sg-s/2ddd0fe91f6037ffb1bce28be0e74d4e
# f = open("build_number.txt", "r")
# parameters["Version"] = [f.read(), "The version number"]

def get_global(name: str):
    if name in parameter_name_conversions.keys():
        name = parameter_name_conversions[name]
    if name in parameters.keys():
        return parameters[name][0]


def set_global(name: str, val):
    global parameters
    if name in parameter_name_conversions.keys():
        name = parameter_name_conversions[name]

    if isinstance(val, np.generic):
        val = val.item()

    if name in ["CLASSES"]:
        # This is for things that cannot be changed
        return

    if name in ["unknowns_clss", "knowns_clss"]:
        if isinstance(val, str):
            parameters[name][0] = [int(y) for y in val.removesuffix("]").removeprefix("[").split(sep=", ")]
        elif isinstance(val, list):
            parameters[name][0] = val
        else:
            raise ValueError(f"{name} global only accepts strings and lists")
        return

    if x in ["var_filtering_threshold"]:
        if isinstance(val, str):
            parameters[name][0] = [float(y) for y in val.removesuffix("]").removeprefix("[").split(sep=",")]
        else:
            parameters[name][0] = float(val)
        return

    if name in parameters.keys():
        if isinstance(val, parameters[name][3]):
            parameters[name][0] = val
        else:
            raise ValueError(f"{name} can only be of type {parameters[name][3]}")


save_as_tensorboard = False
datasetRandomOffset = True
dataparallel = True
use_saved_packets = True
