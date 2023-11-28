from torch import nn
import torch
from torch.nn import functional as F
import os
from tqdm import tqdm
import time

#  # #  user defined functions
import Config
from EndLayer import EndLayers
import helperFunctions
import outputDataObject
import Dataload
import Distance_Types

import numpy as np
# from sklearn.metrics import (precision_score, recall_score, f1_score)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ModdedParallel(nn.DataParallel):
    """
    If the default torch DataParallel cannot find an atribute than it tries to get it from a contained module.
    """
    #  From https://github.com/pytorch/pytorch/issues/16885# issuecomment-551779897
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class AttackTrainingClassification(nn.Module):
    """This is the Default Model for the project"""
    def __init__(self, mode="Soft", numberOfFeatures=1504):
        super().__init__()
        self.maxpooling = [4, 2]
        self.convolutional_channels = [32, 64]

        # This is the length of the packets in the dataset we are currently using.
        self.fullyConnectedStart = ((numberOfFeatures) / (self.maxpooling[0]) // 1) - 1
        self.fullyConnectedStart = ((self.fullyConnectedStart) / (self.maxpooling[1]) // 1) - 1
        self.fullyConnectedStart *= self.convolutional_channels[-1]
        self.fullyConnectedStart = int(self.fullyConnectedStart)

        # There are 15 classes
        numClasses = Config.parameters["CLASSES"][0]
        if Config.parameters['Dataloader_Variation'][0] == "Old_Cluster":
            numClasses = numClasses * 32

        # These are for DOC, it has a special model structure. Because of that we allow it to override what we have.
        if Config.parameters['OOD Type'][0] == "DOC":
            class DOC_Module(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.DOC_kernels = nn.ModuleList()
                    for x in Config.DOC_kernels:
                        self.DOC_kernels.append(nn.Conv1d(1, 32, x, device="cuda" if torch.cuda.is_available() else "cpu"))

                def forward(self, input):
                    return torch.concat([alg(input).max(dim=1)[0] for alg in self.DOC_kernels], dim=-1)

            self.fullyConnectedStart = 0
            self.fullyConnectedStart -= np.array([x - 1 for x in Config.DOC_kernels]).sum()
            self.fullyConnectedStart += numberOfFeatures * len(Config.DOC_kernels)

        # This menu switches between the diffrent options for activation functions.
        self.activation = nn.ReLU()
        activations = {"Sigmoid": nn.Sigmoid(), "Tanh": nn.Tanh(), "Leaky": nn.LeakyReLU(), "Elu": nn.ELU(), "PRElu": nn.PReLU(device=device), "Softplus": nn.Softplus(), "Softmax": nn.Softmax(dim=1)}
        self.activation = activations[Config.parameters["Activation"][0]] if Config.parameters["Activation"][0] in activations.keys() else self.activation

        # We use two normal fully connected layers after the CNN specific layers (or substiute layers)
        self.fc1 = nn.Linear(self.fullyConnectedStart, Config.parameters["Nodes"][0], device=device)
        self.fc2 = nn.Linear(Config.parameters["Nodes"][0], numClasses, device=device)

        self.addedLayers = torch.nn.Sequential()
        # If the config says to add more layers, that is done here.
        for x in range(Config.parameters["Number of Layers"][0]):
            self.addedLayers.append(torch.nn.Linear(Config.parameters["Nodes"][0], Config.parameters["Nodes"][0], device=device))
            self.addedLayers.append(self.activation)

        #  self.COOL = nn.Linear(256, 15*n)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(int(Config.parameters["Dropout"][0]))

        self.end = EndLayers(numClasses, type="Soft", cutoff=Config.parameters["threshold"][0])
        self.batchnum = 0
        self.storeReset()
        # COOL neeeds its own outputl layer.
        self.COOL = nn.Linear(Config.parameters["Nodes"][0], numClasses * self.end.DOO, device=device)

        self.los = False
        self.end.end_type = mode
        self.batch_fdHook = None

        self.sequencePackage = nn.Sequential()
        if self.end.end_type == "DOC":
            self.sequencePackage.append(DOC_Module())

        self.sequencePackage.append(self.flatten)
        self.sequencePackage.append(self.fc1)
        self.sequencePackage.append(self.activation)
        self.sequencePackage.append(self.addedLayers)
        self.sequencePackage.append(self.dropout)
        if self.end.end_type != "COOL":
            self.sequencePackage.append(self.fc2)
        else:
            self.sequencePackage.append(self.COOL)

        if Config.dataparallel:
            self.sequencePackage = nn.DataParallel(self.sequencePackage)

        self.batch_saves_identifier = "No Identification Set"
        if False:
            # I am wondering if passing the direct feature vectors to the last layer will help identify specific points,
            #  such as port id numbers.
            # The thought is that the port id numbers get distorted over the course of the model and need to be re-added later.
            self.fc2 = nn.Linear(Config.parameters["Nodes"][0] + numberOfFeatures, numClasses, device=device)
            self.COOL = nn.Linear(Config.parameters["Nodes"][0] + numberOfFeatures, numClasses * self.end.DOO, device=device)

    # Specify how the data passes in the neural network
    def forward(self, x_before_model: torch.Tensor):
        """Runs the model through all the standard layers

        also uses the Compettitive Overcomplete Output Layer alternative layer if the setting is for COOL.
        """
        #  x = to_device(x, device)
        x_before_model = x_before_model.float()
        x = x_before_model.unsqueeze(1)

        x = self.sequencePackage(x)
        return x

    def fit(self, epochs, lr, train_loader, val_loader, opt_func, measurement=None):
        """
        Trains the model on the train_loader and evaluates it off of the val_loader. Also it stores all of the results in model.store.
        It also generates a new line of ScoresAll.csv that stores all of the data for this model. (note: if you are running two threads at once the data will be overriden)

        parameters:
            epochs- the number of epochs to run the training for
            lr- the learning rate to start at, the schedular will change the learning rate over time and can be set in the Config
            train_loader- the torch dataloader to train the model on
            test_loader- Unused but this should be the data that you want to do the final test of the model on, also in the form of a torch dataloader
            val_loader- the model is tested every epoch against this data
            opt_func- the optimization function to use

        returns:
            history- a list of tuples, each containing:
                val_loss - the loss from the validation stage
                val_acc - the accuract from the validation  stage. Note: this accuracy is not used in the save.
                epoch - the epoch that this data was taken
                train_loss - the average training loss per batch of this epoch
        """
        if Config.parameters["attemptLoadModel"][0] == 1:
            startingEpoch = self.loadPoint("Saves/models")
            # If it cannot find a model to load because of some error, the epoch number starts at -1 to avoid overwriting a possilby working model
        else:
            startingEpoch = 0
        history = []
        optimizer = opt_func(self.parameters(), lr)
        if isinstance(Config.parameters["SchedulerStep"][0], float) and Config.parameters["SchedulerStep"][0] != 0:
            sch = torch.optim.lr_scheduler.StepLR(optimizer, Config.parameters["SchedulerStepSize"][0], Config.parameters["SchedulerStep"][0])
        else:
            sch = None
        self.los = helperFunctions.LossPerEpoch("TestingDuringTrainEpochs.csv")
        #  torch.cuda.empty_cache()
        if epochs > 0:
            with tqdm(range(epochs), desc="Running epochs ") as tqdmEpoch:
                for epoch in tqdmEpoch:
                    self.end.resetvals()
                    self.storeReset()
                    if self.batch_fdHook is not None and self.batch_fdHook.rm is not None:
                        self.batch_fdHook.rm.remove()
                    self.batch_fdHook = Distance_Types.forwardHook()
                    self.batch_fdHook.rm = self.flatten.register_forward_hook(self.batch_fdHook)

                    #  Training Phase
                    self.train()
                    train_losses = []
                    num = 0
                    for batch in train_loader:
                        self.train()
                        self.batch_fdHook.class_vals = batch[1][:, 0]
                        # print("Batch")
                        #  batch = to_device(batch, device)
                        #  batch = DeviceDataLoader(batch, device)
                        loss = self.training_step(batch)

                        train_losses.append(loss.detach())
                        self.end.trainMod(batch, self)
                        # print(loss)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        num += 1

                    self.batch_fdHook.rm.remove()

                    if sch is not None:
                        sch.step()

                    # Validation phase
                    result = self.evaluate(val_loader)

                    if epoch > epochs - 5 or result['val_acc'] > 0.7:
                        self.savePoint("Saves/models", epoch + startingEpoch)

                    result['train_loss'] = torch.stack(train_losses).mean().item()
                    #  if epoch%epoch_record_rate == 0:
                    #      measurement(f"Epoch{epoch+startingEpoch} loss", result['train_loss'])
                    # This seems like it would be slow, TODO: Write this better.
                    if hasattr(measurement, "writer") and measurement.writer is not None:
                        for x in result.keys():
                            measurement.writer.add_scalar(f"Epoch {x}", result[x], global_step=epoch + startingEpoch)
                    result["epoch"] = epoch + startingEpoch
                    tqdmEpoch.set_postfix({"Epoch": epoch + startingEpoch, "train_loss": result['train_loss'], "val_loss": result['val_loss'], "val_acc": result['val_acc']})
                    #  self.epoch_end(epoch+startingEpoch, result)
                    # print("result", result)

                    history.append(result)
                    # self.los.collect(measurement)
                    self.epoch = epoch + startingEpoch
        else:
            #  Validation phase
            if Config.parameters["attemptLoadModel"][0] == 0:
                # don't double load
                epoch = self.loadPoint("Saves/models")
            result = self.evaluate(val_loader)
            result['train_loss'] = -1
            self.epoch_end(epoch, result)
            # print("result", result)
            history.append(result)
            self.epoch = epoch
        return history

    def training_step(self, batch):
        """Preforms a step for training the model but does not begin backpropigation

        Parameters:
            Batch- a torch dataloader batch. Which is a tuple of tensors.

        Returns:
            Loss- a torch loss that signifies how far away from the expected targets the model got.

        """
        data, labels = batch
        # Our labels have two values per line so that we can tell what the unknowns are.
        labels = labels[:, 0]    # Select the data we want not the metadata
        out = self(data)  # Generate predictions
        labels = self.end.labelMod(labels)

        # Not sure if this is nessiary.
        if self.end == "DOC":
            out = nn.Sigmoid()(out)

        if labels.device != out.device:
            labels = labels.to(out.device)

        #  out = DeviceDataLoader(out, device)
        loss = F.cross_entropy(out, labels)  # Calculate loss
        # torch.cuda.empty_cache()
        #  print("loss from training step ... ", loss)
        return loss

    def evaluate_batch(self, batch):
        """
        Takes a batch from the validation loader and evaluates it using the endlayer.

        parameters:
            batch- a single batch from a torch dataloader

        returns:
            dictionary of:
                val_loss - the loss from the validation stage
                val_acc - the accuract from the validation  stage. Note: this accuracy is not used in the save.

        """
        self.eval()
        t = time.time()
        t
        # self.savePoint("test", phase=Config.helper_variables["phase"])
        data, labels_extended = batch
        self.batchnum += 1
        labels = labels_extended[:, 0]

        out = self(data)  # Generate predictions
        zeross = torch.zeros(len(out), 1, device=device)
        loss = F.cross_entropy(torch.cat((out, zeross), dim=1), labels)  # Calculate loss
        out = self.end(out, labels).to(labels.device)  # <----Here is where it is using Softmax TODO: make this be able to run all of the versions and save the outputs.
        # loss = F.cross_entropy(torch.cat((out, zeross), dim=1), labels)  #  Calculate loss
        #  out = self.end.endlayer(out, labels, type="Open")
        #  out = self.end.endlayer(out, labels, type="Energy")

        #  Y_pred = out
        #  Y_test = labels
        #  print("y-test from validation", Y_test)
        #  print("y-pred from validation", Y_pred)

        # This is just for datacollection.
        if self.los:
            if self.end.end_type == "DOC" or self.end.end_type == "COOL":
                self.los.addloss(out, labels)
            else:
                self.los.addloss(torch.argmax(out, dim=1), labels)

        out = out.to(device, non_blocking=True)
        acc = self.accuracy(out, labels_extended)  # Calculate accuracy
        # print("validation accuracy: ", acc)

        return {'val_loss': loss.detach(), 'val_acc': acc}

    def generateDataObject(self, batch: torch.Tensor, generateStability=True) -> outputDataObject.outputDataUpdateObject:
        """
        This is the main way of generating output data from the model, it outputs a outputDataObject for updating the frontend console.

        Parameters:
            batch - Torch tensor containing the data to pass through the model.
                This data should be in the form of: (IxF) where I is the number of instances of the data and F is the number of features.
            generateStability - Booliean value stating if datashiftFactor should be calculated.
        """
        if not isinstance(batch, torch.Tensor):
            batch = Dataload.pandas_to_tensor(batch)

        new_data = outputDataObject.outputDataUpdateObject()
        self.eval()
        out_logits = self(batch)  # Run the model without the endlayer
        out = self.end(out_logits).cpu()  # Run the endlayer
        # Find Argmax:
        if out.ndim == 2:
            predictions = torch.argmax(out, dim=1)
        else:
            # DOC already applies an argmax equivalent so we do not apply one here.
            predictions = out

        new_data.predictions_numerical = predictions.numpy()
        index_to_class = Dataload.CLASSLIST.copy()
        index_to_class[len(index_to_class)] = "Unknown"
        new_data.predictions_string = [index_to_class[x] for x in new_data.predictions_numerical]
        new_data.prediction_confidence = torch.softmax(out_logits, dim=1).max(dim=1)[0]

        # This bit of code generates the stability metrics (Incomplete)
        if generateStability:
            self.batch_fdHook.class_vals = predictions
            self.batch_fdHook.rm = self.flatten.register_forward_hook(self.batch_fdHook)
            self(batch)
            self.batch_fdHook.rm.remove()
            new_data.datashiftFactor = np.array([self.batch_fdHook.distances[x].detach() for x in self.batch_fdHook.distances.keys()]).sum()

        assert isinstance(predictions, torch.Tensor), "Model Output Not torch Tensor, should not be possible"
        bincounts = predictions.bincount(minlength=Config.parameters["CLASSES"][0] + 1)
        new_data.attacks = {Dataload.CLASSLIST[y + 1]: x.item() for y, x in enumerate(bincounts[1:Config.parameters["CLASSES"][0]])}
        new_data.num_packets = bincounts.sum().item()
        # This is temparary:
        for y in torch.softmax(out_logits[predictions == Config.parameters["CLASSES"][0]], dim=1):
            new_data.unknowns.append([Dataload.CLASSLIST[z] for z, x in enumerate(y) if x > 0.2])

        return new_data

    @torch.no_grad()
    def evaluate(self, testset):
        """
        Evaluates the given dataset on this model.

        parameters:
            torch dataset to iterate through.

        returns:
            A dictionary of all of the mean values from the run consisting of:
                val_loss - the loss from the validation stage
                val_acc - the accuract from the validation  stage
        """
        self.eval()
        self.batchnum = 0
        outputs = [self.evaluate_batch(batch) for batch in testset]
        return self.evaluation_epoch_end(outputs)

    def accuracy(self, outputs: torch.Tensor, labels):
        """
        Finds the final accuracy of the batch and saves the predictions and true values to model.store

        parameters:
            outputs- a torch Tensor containing all of the outputs from the models forward pass.
            labels- a torch Tensor containing all of the true and tested against values corresponding to the output results
                labels[:, 0]- should be the values that you are training the model to get
                labels[:, 1:]- should be the true values or other metadata that you want to store associated with a datapoint.

        returns:
            A dictionary of all of the mean values from the run.
        """
        if outputs.ndim == 2:
            preds = torch.argmax(outputs, dim=1)
        else:
            # DOC already applies an argmax equivalent so we do not apply one here.
            preds = outputs
        #  print("preds from accuracy", preds)
        #  print("labels from accuracy", labels)
        #  Y_Pred.append(preds.tolist()[:])
        #  Y_test.append(labels.tolist()[:])
        #  preds = torch.tensor(preds)

        # Find out if something failed, if it did get no accuracy
        if outputs.max() == 0:
            return torch.tensor(0.0)

        # First is the guess, second is the actual class and third is the class to consider correct.
        self.store = torch.cat((self.store[0], preds)), torch.cat((self.store[1], labels[:, 1])), torch.cat((self.store[2], labels[:, 0]))
        return torch.tensor(torch.sum(preds == labels[:, 0]).item() / len(preds))
        #  def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

    def evaluation_epoch_end(self, outputs):
        """
        Takes the output of each epoch and takes the mean values. Returns a dictionary of those mean values.

        returns:
            dictionary of:
                val_loss - the loss from the validation stage
                val_acc - the accuract from the validation  stage
        """
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        """
        prints the results using the epoch
        """
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(epoch,
                                                                                         result['train_loss'],
                                                                                         result['val_loss'],
                                                                                         result['val_acc']))

    def savePoint(net, path: str, epoch=0):
        """
        Saves the model

        parameters:
            path- path to the folder to store the model in.
            epoch- the number of epochs the model has trained.

        """
        if Config.unit_test_mode:
            return
        if not os.path.exists(path):
            os.mkdir(path)
        to_save = {
            "model_state": net.state_dict(),
            "parameter_keys": list(Config.parameters.keys()),
            "parameters": Config.parameters,
            "CLASSLIST": Dataload.CLASSLIST,
            "LISTCLASS": Dataload.LISTCLASS
        }
        if net.batch_fdHook is not None:
            to_save["batchSaveClassMeans"] = net.batch_fdHook.means
        to_save["parameter_keys"].remove("optimizer")
        to_save["parameter_keys"].remove("Unknowns")
        torch.save(to_save, path + f"/Epoch{epoch:03d}{Config.parameters['OOD Type'][0]}")

        if epoch >= 5:
            oldPath, epoch = AttackTrainingClassification.findloadPath(path, epoch=epoch - 5)
            if os.path.exists(oldPath):
                os.remove(oldPath)

    def loadPoint(net, path=None, deleteOld=False):
        """
        Loads the most trained model from the path. Note: will break if trying to load a model with different configs.

        parameters:
            path- path to the folder where the models are stored.

        returns:
            epochFound - the number of epochs the model that was found has run for.
        """

        if path is None:
            pathFound, epochFound = AttackTrainingClassification.findloadPath(path)
        else:
            pathFound, epochFound = (path, 0)
        loaded = torch.load(pathFound, map_location=device)

        print(f"Loaded  model from {pathFound}")

        # # Count the classes
        if all([x in Dataload.CLASSLIST.keys() for x in loaded["CLASSLIST"].keys()]) and all([loaded["CLASSLIST"][x] == Dataload.CLASSLIST[x] for x in loaded["CLASSLIST"].keys()]):
            print("Model has identical classes")
        else:
            Dataload.CLASSLIST = loaded["CLASSLIST"]
            Dataload.LISTCLASS = loaded["LISTCLASS"]
            Config.recountclasses(Dataload.CLASSLIST)
            print(f"CLASSES have changed, there are now {Config.parameters['CLASSES'][0]} classes")
            net = net.__class__()

        for x in loaded["parameter_keys"]:

            #  assert x in Config.parameters.keys() # Make sure that the model loaded actually has all of the needed values
            if x in Config.parameters.keys() and loaded["parameters"][x][0] != Config.parameters[x][0]:
                if x not in ["model", "CLASSES", "Degree of Overcompleteness", "Number of Layers", "Nodes"]:
                    print(f"Warning: {x} has been changed from when model was created")
                else:
                    print(f"Critital mismatch for model {x} is different from loaded version. No load can occur")
                    return -1
        for x in loaded["parameters"]["Unknowns_clss"][0]:
            if x not in Config.parameters["Unknowns_clss"][0]:
                print(f"Warning: Model trained with {x} as an unknown.")
        net.load_state_dict(loaded["model_state"])
        if "batchSaveClassMeans" in loaded.keys():
            net.batch_fdHook = Distance_Types.forwardHook()
            net.batch_fdHook.means = loaded["batchSaveClassMeans"]

        return epochFound

    @staticmethod
    def findloadEpoch(path="Saves/models"):
        """
        Finds the highest existing epoch save for this model type.
        returns -1 if none exists
        """
        if not os.path.exists(path):
            os.mkdir(path)
        i = 999
        # epochFound = -1
        for i in range(1000, -1, -1):
            if os.path.exists(path + f"/Epoch{i:03d}{Config.parameters['OOD Type'][0]}"):
                return i
        return -1

    @staticmethod
    def findloadPath(path="Saves/models", epoch=None):
        if not os.path.exists(path):
            os.mkdir(path)
        if (epoch is not None):
            epochFound = epoch
        else:
            epochFound = AttackTrainingClassification.findloadEpoch(path)
        if epochFound == -1:
            print("No model to load found.")
            return "", -1
        return path + f"/Epoch{epochFound:03d}{Config.parameters['OOD Type'][0]}", epochFound

    def storeReset(self):
        """
        Resets the storage for the model.
        """
        self.store = torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)


class expand_bitPackets(nn.Module):
    def __init__(self, lenExpand=1500):
        super().__init__()
        self.length_to_expand = lenExpand
        self.cnn = nn.Conv2d(1, 1, 50)

    def forward(self, input: torch.Tensor):
        toMod = input[:self.length_to_expand].int()
        # shape = toMod.shape
        toMod = torch.concat([binary(toMod // (32**x), bits=32) for x in range(4)], dim=-1)
        assert isinstance(toMod, torch.Tensor)
        toMod = self.cnn(toMod.float())
        toMod.squeeze()
        input[:self.length_to_expand] = toMod
        return input


#  Binary from: https://stackoverflow.com/a/63546308 <- I like how clean this code is
def binary(x, bits):
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


class shift_instance(nn.Module):
    def __init__(self, amount=0):
        super().__init__()
        self.amount = amount

    def forward(self, input: torch.Tensor):
        input[:-self.amount] = input[self.amount:]
        input[-self.amount:] = torch.zeros_like(input[-self.amount:])
        return input


class Conv1DClassifier(AttackTrainingClassification):
    def __init__(self, mode="Soft", numberOfFeatures=1504):
        super().__init__(mode, numberOfFeatures)
        self.layer1 = nn.Sequential(
            # expand_bitPackets(),
            nn.Conv1d(1, self.convolutional_channels[0], 3, device=device),
            self.activation,
            nn.MaxPool1d(self.maxpooling[0]),
            nn.Dropout(int(Config.parameters["Dropout"][0])))
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.convolutional_channels[0], self.convolutional_channels[1], 3, device=device),
            self.activation,
            nn.MaxPool1d(self.maxpooling[1]),
            nn.Dropout(int(Config.parameters["Dropout"][0])))

        sequencePackage = nn.Sequential()

        sequencePackage.append(self.layer1)
        sequencePackage.append(self.layer2)
        if self.end.end_type != "DOC":
            if Config.dataparallel:
                sequencePackage.append(self.sequencePackage.module)
                self.sequencePackage = nn.DataParallel(sequencePackage)
            else:
                sequencePackage.append(self.sequencePackage)
                self.sequencePackage = sequencePackage


class FullyConnected(AttackTrainingClassification):
    def __init__(self, mode="Soft", numberOfFeatures=1504):
        super().__init__(mode, numberOfFeatures)
        self.layer1 = nn.Sequential(
            # Sorry about the big block of math, trying to calculate how big the convolutional tensor is after the first layer
            nn.Linear(numberOfFeatures, int(self.convolutional_channels[0] * (((numberOfFeatures) / (self.maxpooling[0]) // 1) - 1)), device=device),
            self.activation,
            nn.Dropout(int(Config.parameters["Dropout"][0])))
        self.layer2 = nn.Sequential(
            nn.Linear(int(self.convolutional_channels[0] * (((numberOfFeatures) / (self.maxpooling[0]) // 1) - 1)), self.fullyConnectedStart, device=device),
            self.activation,
            nn.Dropout(int(Config.parameters["Dropout"][0])))

        sequencePackage = nn.Sequential()

        sequencePackage.append(self.layer1)
        sequencePackage.append(self.layer2)
        if self.end.end_type != "DOC":
            if Config.dataparallel:
                sequencePackage.append(self.sequencePackage.module)
                self.sequencePackage = nn.DataParallel(sequencePackage)
            else:
                sequencePackage.append(self.sequencePackage)
                self.sequencePackage = sequencePackage


def train_model(model: AttackTrainingClassification):
    train, test, val = Dataload.checkAttempLoad()
    new_data = Dataload.savedPacketDataset()
    torch.utils.data.ConcatDataset([train, new_data])

    training = Dataload.DataLoader(train, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    testing = Dataload.DataLoader(test, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)
    validation = Dataload.DataLoader(val, Config.parameters["batch_size"][0], shuffle=True, num_workers=0, pin_memory=False)

    testing
    validation

    model.end.prepWeibull(training, torch.device('cpu'), model)
    model.fit(Config.parameters["num_epochs"][0], Config.parameters["learningRate"][0], training, validation, opt_func=torch.optim.Adam)
