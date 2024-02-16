from torch import nn
import torch
from torch.nn import functional as F
import os
from tqdm import tqdm
import time
from sender import sender

### user defined functions
import Config
from EndLayer import EndLayers
import GPU
import FileHandling
import helperFunctions
import Distance_Types

import numpy as np
from sklearn.metrics import (precision_score, recall_score, average_precision_score,accuracy_score,f1_score)

device = GPU.get_default_device()

class ModdedParallel(nn.DataParallel):
    """
    If the default torch DataParallel cannot find an atribute than it tries to get it from a contained module.
    """
    # From https://github.com/pytorch/pytorch/issues/16885#issuecomment-551779897
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        


class AttackTrainingClassification(nn.Module):
    """This is the Default Model for the project"""
    def __init__(self,mode="Soft",numberOfFeatures=1504):
        super().__init__()
        self.maxpooling = [4,2]
        self.convolutional_channels = [32,64]
        

        # This is for doing a convolution of the bits
        self.bitpack = expand_bitPackets(numberOfFeatures,device=device)

        #This is the length of the packets in the dataset we are currently using.
        self.fullyConnectedStart = ((numberOfFeatures)/(self.maxpooling[0])//1)-1
        self.fullyConnectedStart =  ((self.fullyConnectedStart)/(self.maxpooling[1])//1)-1
        self.fullyConnectedStart *= self.convolutional_channels[-1]
        self.fullyConnectedStart = int(self.fullyConnectedStart)

        #There are 15 classes
        numClasses = Config.parameters["CLASSES"][0]
        if Config.parameters['Dataloader_Variation'][0] == "Old_Cluster":
            numClasses = numClasses*32
        
        #These are for DOC, it has a special model structure. Because of that we allow it to override what we have.
        if Config.parameters['OOD Type'][0] == "DOC":
            class DOC_Module(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.DOC_kernels = nn.ModuleList()
                    for x in Config.DOC_kernels:
                        self.DOC_kernels.append(nn.Conv1d(1, 32, x,device="cuda" if torch.cuda.is_available() else "cpu"))
                def forward(self,input):
                    return torch.concat([alg(input).max(dim=1)[0] for alg in self.DOC_kernels],dim=-1)
            self.fullyConnectedStart = 0
            self.fullyConnectedStart -= np.array([x-1 for x in Config.DOC_kernels]).sum()
            self.fullyConnectedStart+= numberOfFeatures*len(Config.DOC_kernels)

        #This menu switches between the diffrent options for activation functions.
        self.activation = nn.ReLU()
        activations = {"Sigmoid": nn.Sigmoid(), "Tanh": nn.Tanh(),"Leaky":nn.LeakyReLU(),"Elu":nn.ELU(),"PRElu":nn.PReLU(device=device),"Softplus":nn.Softplus(),"Softmax":nn.Softmax(dim=1)}
        self.activation = activations[Config.parameters["Activation"][0]] if Config.parameters["Activation"][0] in activations.keys() else self.activation


        #We use two normal fully connected layers after the CNN specific layers (or substiute layers)
        self.fc1 = nn.Linear(self.fullyConnectedStart, Config.parameters["Nodes"][0],device=device)
        self.fc2 = nn.Linear(Config.parameters["Nodes"][0], numClasses,device=device)

        class RecordBatchVals_afterFC1(nn.Module):
            def __init__(self,activation):
                super().__init__()
                self.activation = activation
            
            def forward(me,x):
                self.batch_saves_fucnt("Difference after Fully_Connected_1",x.max().item()-x.min().item())
                x = self.activation(x)
                self.batch_saves_fucnt("Average of layer Fully_Connected_1 Node 0",x.permute(1,0).mean(dim=1)[0].item())
                self.batch_saves_fucnt("Average of layer Fully_Connected_1 Total",x.mean().item())
                return x


        self.addedLayers = torch.nn.Sequential()
        #If the config says to add more layers, that is done here.
        for x in range(Config.parameters["Number of Layers"][0]):
            self.addedLayers.append(torch.nn.Linear(Config.parameters["Nodes"][0],Config.parameters["Nodes"][0],device=device))
            self.addedLayers.append(self.activation)

        # self.COOL = nn.Linear(256, 15*n)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(int(Config.parameters["Dropout"][0]))

        self.end = EndLayers(numClasses, type="Soft", cutoff=Config.parameters["threshold"][0])
        self.end_soft = EndLayers(numClasses, type="SoftThresh", cutoff=Config.parameters["threshold"][0])
        self.batchnum = 0
        self.storeReset()
        #COOL neeeds its own outputl layer.
        self.COOL = nn.Linear(Config.parameters["Nodes"][0], numClasses*self.end.DOO,device=device)

        self.los = False
        self.end.end_type = mode
        self.keep_batch_saves = False
        self.batch_fdHook = None



        self.sequencePackage = nn.Sequential()
        if self.end.end_type=="DOC":
            self.sequencePackage.append(DOC_Module())
        
        
        self.sequencePackage.append(self.flatten)
        self.sequencePackage.append(self.fc1)
        if self.keep_batch_saves:
            #I just realized this never activates. I made this 3 months ago.
            self.sequencePackage.append(RecordBatchVals_afterFC1(self.activation))
        else:
            self.sequencePackage.append(self.activation)
        self.sequencePackage.append(self.addedLayers)
        self.sequencePackage.append(self.dropout)
        if self.end.end_type!="COOL":
            self.sequencePackage.append(self.fc2)
        else:
            self.sequencePackage.append(self.COOL)
        
        if Config.dataparallel:
            self.sequencePackage = nn.DataParallel(self.sequencePackage)


        self.batch_saves_identifier = "No Identification Set"
        if False:
            #I am wondering if passing the direct feature vectors to the last layer will help identify specific points, 
            # such as port id numbers.
            #The thought is that the port id numbers get distorted over the course of the model and need to be re-added later.
            self.fc2 = nn.Linear(Config.parameters["Nodes"][0]+numberOfFeatures, numClasses,device=device)
            self.COOL = nn.Linear(Config.parameters["Nodes"][0]+numberOfFeatures, numClasses*self.end.DOO,device=device)


    # Specify how the data passes in the neural network
    def forward(self, x_before_model: torch.Tensor):
        """Runs the model through all the standard layers
        
        also uses the Compettitive Overcomplete Output Layer alternative layer if the setting is for COOL.
        """
        # x = to_device(x, device)
        x_before_model = x_before_model.float()
        x = x_before_model.unsqueeze(1)
        if Config.parameters["Experimental_bitConvolution"][0] == 1:
            x = self.bitpack(x)
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
                train_loss - the average training loss per batch oncostagliolaf this epoch
        """
        if measurement is None:
            measurement = FileHandling.Score_saver()
        if Config.parameters["attemptLoadModel"][0] == 1:
            startingEpoch = self.loadPoint("Saves/models")
            #If it cannot find a model to load because of some error, the epoch number starts at -1 to avoid overwriting a possilby working model
        else:
            startingEpoch = 0
        history = []
        optimizer = opt_func(self.parameters(), lr)
        if isinstance(Config.parameters["SchedulerStep"][0],float) and Config.parameters["SchedulerStep"][0] !=0:
            sch = torch.optim.lr_scheduler.StepLR(optimizer, Config.parameters["SchedulerStepSize"][0], Config.parameters["SchedulerStep"][0])
        else:
            sch = None
        self.los = helperFunctions.LossPerEpoch("TestingDuringTrainEpochs.csv")
        # torch.cuda.empty_cache()a

        if epochs > 0:
            with tqdm(range(epochs),desc="Running epochs ") as tqdmEpoch:
                for epoch in tqdmEpoch:
                    self.end.resetvals()
                    self.storeReset()
                    # Training Phase
                    self.train()
                    train_losses = []
                    num = 0
                    for batch in train_loader:
                        if self.keep_batch_saves:
                            self.batch_saves_start()
                            self.batch_saves_fucnt("Kind","Training")
                        self.train()
                        #print("Batch")
                        # batch = to_device(batch,device)
                        # batch = DeviceDataLoader(batch, device)
                        loss = self.training_step(batch)

                        #FileHandling.write_batch_to_file(loss, num, self.end.type, "train")
                        train_losses.append(loss.detach())
                        self.end.trainMod(batch,self)
                        #print(loss)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        num += 1

                    if not sch is None:
                        sch.step()
                    

                    # Validation phase
                    result = self.evaluate(val_loader)

                    if epoch > epochs-5 or result['val_acc'] > 0.7:
                        self.savePoint(f"Saves/models", epoch+startingEpoch)

                    result['train_loss'] = torch.stack(train_losses).mean().item()
                    # if epoch%epoch_record_rate == 0:
                    #     measurement(f"Epoch{epoch+startingEpoch} loss",result['train_loss'])
                    #This seems like it would be slow, TODO: Write this better.
                    if hasattr(measurement,"writer") and measurement.writer is not None:
                        for x in result.keys():
                            measurement.writer.add_scalar(f"Epoch {x}",result[x],global_step=epoch+startingEpoch)
                    result["epoch"] = epoch+startingEpoch
                    tqdmEpoch.set_postfix({"Epoch":epoch+startingEpoch, "train_loss": result['train_loss'], "val_loss": result['val_loss'], "val_acc": result['val_acc']})
                    # self.epoch_end(epoch+startingEpoch, result)
                    #print("result", result)

                    history.append(result)
                    self.los.collect(measurement)
                    self.epoch = epoch+startingEpoch
        else:
            # Validation phase
            if Config.parameters["attemptLoadModel"][0] == 0:
                # don't double load
                epoch = self.loadPoint("Saves/models")
            else:
                epoch = startingEpoch
            result = self.evaluate(val_loader)
            result['train_loss'] = -1
            self.epoch_end(epoch, result)
            #print("result", result)
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
        #Our labels have two values per line so that we can tell what the unknowns are.
        labels = labels[:,0]    #Select the data we want not the metadata
        out = self(data)  # Generate predictions
        labels = self.end.labelMod(labels)

        
        #Not sure if this is nessiary. 
        if self.end == "DOC":
            out = nn.Sigmoid()(out)
        
        
        

        # out = DeviceDataLoader(out, device)
        loss = F.cross_entropy(out, labels)  # Calculate loss
        #torch.cuda.empty_cache()
        # print("loss from training step ... ", loss)
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
        #self.savePoint("test", phase=Config.helper_variables["phase"])
        data, labels_extended = batch
        self.batchnum += 1
        labels = labels_extended[:,0]
        # cumulative_logit_data = FileHandling.Score_saver(path="Distances.csv")
        rm = []
        # rm.append(self.fc1.register_forward_hook(lambda x,y,z: item_logit_data.storeItems(y[0])))
        # rm.append(self.fc1.register_forward_hook(lambda x,y,z: cumulative_logit_data("Average standard Div",torch.mean(torch.std(y[0])).item())))
        # rm.append(self.fc1.register_forward_hook(lambda x,y,z: cumulative_logit_data("Average mean",torch.mean(y[0]).item())))
        # rm.append(self.fc1.register_forward_hook(lambda x,y,z: cumulative_logit_data(f"Item Count Class", labels.bincount(minlength=Config.parameters['CLASSES'][0]).numpy(),recursiveList=1)))
        # rm = self.fc1.register_forward_hook(lambda x,y,z: print(f"Average standard Div: {torch.mean(torch.std(y[0]))}, Average mean {torch.mean(y[0])}, Labels Dist{labels.bincount(minlength=Config.parameters['CLASSES'][0]).numpy()}"))

        if self.keep_batch_saves:
            self.batch_saves_start()
            self.batch_saves_fucnt("Kind","Testing")
            # removeHandle = self.sequencePackage.module.register_module_forward_hook(self.batch_fdHook())

        #Data - input data
        out_pre_endlayer = self(data)  # Generate predictions
        if Config.parameters["ItemLogitData"][0] == 1:
            item_logit_data = FileHandling.items_with_classes_record(labels_extended)
            item_logit_data.storeItems(out_pre_endlayer)
        
        [x.remove() for x in rm]
        #zeross = GPU.to_device(torch.zeros(len(out),1),device)
        zeross = GPU.to_device(torch.zeros(len(out_pre_endlayer),1),device)
        loss = F.cross_entropy(torch.cat((out_pre_endlayer,zeross),dim=1), labels)  # Calculate loss
        out_post_endlayer = self.end(out_pre_endlayer, labels).to(labels.device)  # <----Here is where it is using Softmax TODO: make this be able to run all of the versions and save the outputs.
        out_soft_threshold = self.end_soft(out_pre_endlayer, labels).to(labels.device)

        if False:
            if Config.parameters["ApplyPrelimSoft"][0] and self.end.end_type not in ["COOL","DOC"]:
                out_post_endlayer[torch.argmax(out_soft_threshold,dim=1)!=Config.parameters["CLASSES"][0]] = out_soft_threshold[[torch.argmax(out_soft_threshold,dim=1)!=Config.parameters["CLASSES"][0]]]

        #loss = F.cross_entropy(torch.cat((out,zeross),dim=1), labels)  # Calculate loss
        # out = self.end.endlayer(out, labels, type="Open")
        # out = self.end.endlayer(out, labels, type="Energy")

        if out_post_endlayer.ndim == 2:
            out_argmax = torch.argmax(out_post_endlayer, dim=1).cpu()
        else:
            #DOC already applies an argmax equivalent so we do not apply one here.
            out_argmax = out_post_endlayer.cpu()

        if Config.parameters["ItemLogitData"][0] == 1:
            item_logit_data.storePredictions(out_argmax)
            item_logit_data.useItems()

        # Y_pred = out
        # Y_test = labels
        # print("y-test from validation",Y_test)
        # print("y-pred from validation", Y_pred)

        #This is just for datacollection.
        if self.los:
            if self.end.end_type == "DOC" or self.end.end_type == "COOL":
                self.los.addloss(out_post_endlayer,labels)
            else:
                self.los.addloss(torch.argmax(out_post_endlayer,dim=1),labels)

        out_post_endlayer = GPU.to_device(out_post_endlayer, device)
        acc = self.accuracy(out_post_endlayer, labels_extended)  # Calculate accuracy
        #FileHandling.write_batch_to_file(loss, self.batchnum, self.end.type, "Saves")
        #print("validation accuracy: ", acc)
        #self.batch_saves_fucnt("Knowns Only F1_Score",f1_score(labels_extended[labels_extended[:, 0] != Config.parameters["CLASSES"][0], 0],out_argmax[labels_extended[:, 0] != Config.parameters["CLASSES"][0]],average="weighted", zero_division=0))
        rah = out_argmax
        mask = rah == 15
        indices_of_true_values = mask.nonzero(as_tuple=False).squeeze()
        #print(indices_of_true_values)
        selected_data = data[indices_of_true_values]
        is_empty = (selected_data.size(0) == 0)
        if(is_empty == False):
            # Convert the tensor to a NumPy array
            selected_data_cpu = data.cpu()
            integer_tensor = selected_data_cpu.to(dtype=torch.int)
            integer_array = integer_tensor.numpy()

            # Convert integers in the second dimension to hexadecimal
            hex_array = np.vectorize(hex)(integer_array).astype(str)
            hex_strings = [''.join(format(int(byte, 16), '02x') for byte in row).rstrip('0') for row in hex_array]
            np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            print(len(hex_strings))


        if False:
            # Get the indices of the maximum values along dimension 1
            argmax_indices = torch.argmax(out_post_endlayer, dim=1)

            # Create a mask for elements numbered 15
            mask_for_15 = (out_post_endlayer == 15).int()  # Since indexing is zero-based, 15 is at index 14

            # Use the mask to access elements numbered 15
            elements_numbered_15 = out_post_endlayer[mask_for_15]
            selected_data = data[mask_for_15.nonzero(as_tuple=False)]
            print(selected_data.shape)



        if self.keep_batch_saves:
            self.batch_saves_fucnt("Identfier",self.batch_saves_identifier)
            if isinstance(self.end.rocData[1],torch.Tensor):
                self.batch_saves_fucnt(f"Average unknown threshold possibilities",self.end.rocData[1].mean().item())
            else:
                self.batch_saves_fucnt(f"Average unknown threshold possibilities",np.array(self.end.rocData[1]).mean())
            self.batch_saves_fucnt("Overall Accuracy",acc.item())
            
            prec = precision_score(labels_extended[:,0].cpu(),out_argmax, labels=[Config.parameters["CLASSES"][0]],average="weighted",zero_division=0)
            rec = recall_score(labels_extended[:,0].cpu(),out_argmax, labels=[Config.parameters["CLASSES"][0]],average="weighted",zero_division=0)
            self.batch_saves_fucnt("Knowns/Unknowns_Precision",prec)
            self.batch_saves_fucnt("Knowns/Unknowns_False Positive Rate",1-prec)
            self.batch_saves_fucnt("Knowns/Unknowns_Recall",rec)
            self.batch_saves_fucnt("Knowns/Unknowns_False Negative Rate",1-rec)
            self.batch_saves_fucnt("Knowns/Unknowns_F1_Score",f1_score(labels_extended[:,0].cpu(),out_argmax, labels=[Config.parameters["CLASSES"][0]],average="weighted",zero_division=0))
            self.batch_saves_fucnt("Total F1_Score",f1_score(labels_extended[:,0].cpu(),out_argmax,average="weighted",zero_division=0))
            if len(labels_extended[labels_extended[:,0]!=Config.parameters["CLASSES"][0],0]) != 0:
                labels_extended = labels_extended.cpu()
                out_argmax = out_argmax.cpu()
                #Mask Example
                self.batch_saves_fucnt("Knowns Only F1_Score",f1_score(labels_extended[labels_extended[:,0]!=Config.parameters["CLASSES"][0],0],out_argmax[labels_extended[:,0]!=Config.parameters["CLASSES"][0]],average="weighted",zero_division=0))
            self.batch_saves_fucnt("Time",time.time()-t)
            # torch.Tensor.bincount(minlength=Config.parameters["CLASSES"][0])
            sampleCounts = labels.bincount(minlength=Config.parameters["CLASSES"][0]+1)
            guessCounts = out_post_endlayer.argmax(dim=-1).bincount(minlength=Config.parameters["CLASSES"][0]+1)
            # for i in range(Config.parameters["CLASSES"][0]):
            #     self.batch_saves_fucnt(f"Samples of class {i}",sampleCounts[i].item())
            #     self.batch_saves_fucnt(f"Guesses of class {i}",guessCounts[i].item())
                # if guessCounts[i].item()!=0:
                #     self.batch_saves_fucnt(f"Samples/Guesses of class {i}",sampleCounts[i].item()/guessCounts[i].item())
            mask = torch.concat([helperFunctions.mask,torch.tensor([False])])
            self.batch_saves_fucnt(f"Samples of known classes",sampleCounts[mask].sum().item())
            self.batch_saves_fucnt(f"Guesses of known classes",guessCounts[mask].sum().item())
            if guessCounts[mask].sum().item()!=0:
                self.batch_saves_fucnt(f"Samples/Guesses of known classes",sampleCounts[mask].sum().item()/guessCounts[mask].sum().item())
            mask = mask==False
            self.batch_saves_fucnt(f"Samples of unknown classes",sampleCounts[mask].sum().item())
            self.batch_saves_fucnt(f"Guesses of unknown classes",guessCounts[mask].sum().item())
            if guessCounts[mask].sum().item()!=0:
                self.batch_saves_fucnt(f"Samples/Guesses of unknown classes",sampleCounts[mask].sum().item()/guessCounts[mask].sum().item())
            if self.end.end_type not in ["COOL","DOC"]:
                self.batch_saves_fucnt("intra_spread_Endlayer",Distance_Types.distance_measures(out_post_endlayer.cpu(),self.batch_fdHook.means["End"],torch.argmax(out_post_endlayer,dim=1).cpu(),Distance_Types.dist_types_dict["intra_spread"]).item())
                self.batch_saves_fucnt("Cosine_dist_Endlayer",Distance_Types.distance_measures(out_post_endlayer.cpu(),self.batch_fdHook.means["End"],torch.argmax(out_post_endlayer,dim=1).cpu(),Distance_Types.dist_types_dict["Cosine_dist"]).item())
                self.batch_saves_fucnt("Euclidean_Distance_Endlayer",Distance_Types.distance_measures(out_post_endlayer.cpu(),self.batch_fdHook.means["End"],torch.argmax(out_post_endlayer,dim=1).cpu(),Distance_Types.dist_types_dict["Euclidean Distance"]).item())

            #Calculating cluster distances
            if False:
                self.batch_fdHook.class_vals = out_argmax
                removeHandle = torch.nn.modules.module.register_module_forward_hook(self.batch_fdHook)
                for distancetype in ["Cosine_dist","intra_spread","Euclidean Distance"]:
                    self.batch_fdHook.distFunct = distancetype
                    self(data)
                    for name in self.batch_fdHook.distances.keys():
                        self.batch_saves_fucnt(f"{self.batch_fdHook.distFunct} distance of {name}",self.batch_fdHook.distances[name].item())
                    self.batch_fdHook.distances = {}
                removeHandle.remove()
            

        return {'val_loss': loss.detach(), 'val_acc': acc}


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

    def accuracy(self, outputs:torch.Tensor, labels):
        """
        Finds the final accuracy of the batch and saves the predictions and true values to model.store

        parameters:
            outputs- a torch Tensor containing all of the outputs from the models forward pass.
            labels- a torch Tensor containing all of the true and tested against values corresponding to the output results
                labels[:,0]- should be the values that you are training the model to get
                labels[:,1:]- should be the true values or other metadata that you want to store associated with a datapoint.
        
        returns:
            A dictionary of all of the mean values from the run.
        """
        if outputs.ndim == 2:
            preds = torch.argmax(outputs, dim=1)
        else:
            #DOC already applies an argmax equivalent so we do not apply one here.
            preds = outputs
        # print("preds from accuracy", preds)
        # print("labels from accuracy", labels)
        # Y_Pred.append(preds.tolist()[:])
        # Y_test.append(labels.tolist()[:])
        # preds = torch.tensor(preds)

        #Find out if something failed, if it did get no accuracy
        if outputs.max() == 0:
            return torch.tensor(0.0)


        #First is the guess, second is the actual class and third is the class to consider correct.
        self.store = torch.cat((self.store[0], preds)), torch.cat((self.store[1], labels[:,1])),torch.cat((self.store[2], labels[:,0]))
        return torch.tensor(torch.sum(preds == labels[:,0]).item() / len(preds))
        # def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):



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
            "parameters": Config.parameters
        }
        if not net.batch_fdHook is None:
            to_save["batchSaveClassMeans"] = net.batch_fdHook.means
        to_save["parameter_keys"].remove("optimizer")
        to_save["parameter_keys"].remove("Unknowns")
        torch.save(to_save, path + f"/Epoch{epoch:03d}{Config.parameters['OOD Type'][0]}.pth")

        oldPath = AttackTrainingClassification.findloadPath(epoch-5,path)
        if os.path.exists(oldPath):
            os.remove(oldPath)

    def loadPoint(net, path: str, deleteOld=True):
        """
        Loads the most trained model from the path. Note: will break if trying to load a model with different configs.

        parameters:
            path- path to the folder where the models are stored.

        returns:
            epochFound - the number of epochs the model that was found has run for.
        """
        if not os.path.exists(path):
            os.mkdir(path)
        epochFound = AttackTrainingClassification.findloadEpoch(path)
        if epochFound == -1:
            print("No model to load found.")
            return -1
        
        pathFound = AttackTrainingClassification.findloadPath(epochFound,path)
        loaded = torch.load(pathFound,map_location=GPU.get_default_device())
        
        print(f"Loaded  model from {pathFound}")
        for x in loaded["parameter_keys"]:

            # assert x in Config.parameters.keys() #Make sure that the model loaded actually has all of the needed values
            if x in Config.parameters.keys() and loaded["parameters"][x][0] != Config.parameters[x][0]:
                if not x in ["model","CLASSES","Degree of Overcompleteness","Number of Layers","Nodes"]:
                    print(f"Warning: {x} has been changed from when model was created")
                else:
                    print(f"Critital mismatch for model {x} is different from loaded version. No load can occur")
                    return -1
        for x in loaded["parameters"]["Unknowns_clss"][0]:
            if not x in Config.parameters["Unknowns_clss"][0]:
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
        i = 999
        epochFound = -1
        for i in range(1000,-1,-1):
            if os.path.exists(path + f"/Epoch{i:03d}{Config.parameters['OOD Type'][0]}.pth"):
                return i
        return -1

    @staticmethod
    def findloadPath(epoch:int, path="Saves/models"):
        return path + f"/Epoch{epoch:03d}{Config.parameters['OOD Type'][0]}.pth"
    
    #This loops through all the thresholds without resetting the model.
    def thresholdTest(net,val_loader,measurement=None):
        """
        This tests the results from val_loader at various thresholds and saves it to scoresAll.csv

        No longer used in favor of the ROC score.
        """
        if measurement is None:
            measurement = FileHandling.Score_saver()
        net.end.end_type = Config.parameters["OOD Type"][0]
        net.end.end_type = net.end.end_type
        net.loadPoint("Saves/models")
        thresh = Config.thresholds
        for y in range(len(thresh)):
            x = thresh[y]
            #reset
            net.end.resetvals()
            net.store = GPU.to_device(torch.tensor([]), device), GPU.to_device(torch.tensor([]), device), GPU.to_device(torch.tensor([]), device)
            net.end.cutoff = x
            
            #evaluate
            net.evaluate(val_loader)

            #get the data
            y_pred,y_true,y_tested_against = net.store

            #evaluate the data
            y_true = y_true.to(torch.int).tolist()
            y_pred = y_pred.to(torch.int).tolist()
            y_tested_against = y_tested_against.to(torch.int).tolist()
            recall = recall_score(y_tested_against,y_pred,average='weighted',zero_division=0)
            precision = precision_score(y_tested_against,y_pred,average='weighted',zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall)
            #save the f1 score
            # FileHandling.create_params_Fscore("",f1,x)
            measurement(f"Threshold {x} Fscore",f1)


    def storeReset(self):
        """
        Resets the storage for the model.
        """
        self.store = GPU.to_device(torch.tensor([]), device), GPU.to_device(torch.tensor([]), device), GPU.to_device(torch.tensor([]), device)
    
    def batchSaveMode(self,function=None):
        """
        This wraps the saving scores so that the values recorded are piped to a specific file.
        """
        if function is None:
            function = FileHandling.Score_saver(path="BatchSaves.csv",newline=False)
        def start():
            function.create_params_All(name="BatchSaves.csv")
            function("Current threshold",self.end.cutoff,fileName="BatchSaves.csv")
        self.batch_saves_start = start
        self.keep_batch_saves = True
        self.batch_saves_fucnt = function
        self.eval()
        

        #get class means for intra spread
        if self.batch_fdHook is None:
            self.batch_fdHook = Distance_Types.forwardHook()
        if self.end.end_type != "COOL":
            if len(self.batch_fdHook.means) == 0:
                print("Recalculating means Starting",flush=True)
                self.batch_fdHook.means["End"] = Distance_Types.class_means_from_loader(self.end.weibulInfo)
                print("Recalculating means Saved",flush=True)
        

class expand_bitPackets(nn.Module):
    def __init__(self,lenExpand=1500, device=torch.cpu):
        """
        Expands the bit packets and then performs a convolution on them before returning them to the same dimentions.
        The theory behind this is that a convolution will extract better information than just treating each byte as its own integer.
        It has not worked so far.
        
        """
        super().__init__()
        self.length_to_expand = lenExpand

        kernel = (50,4)
        self.cnn = nn.Conv2d(1,20,kernel,device="cpu")

        #equations from: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        #Simplified for our purposes
        Hout = (lenExpand - (kernel[0]-1)) // 1
        Wout = (8 - (kernel[1]-1)) // 1

        self.fc = nn.Linear(Hout*Wout*20, lenExpand,device=device)

    def forward(self, input:torch.Tensor):
        #Gets  the bits out from the packet header information
        toMod = input[:, :, :self.length_to_expand]

        #the apply function does not work with gradients so we need to do everything inside of a nograd block
        with torch.no_grad():
            toMod = toMod.clone().cpu()
            # Integer to binary from here: https://www.geeksforgeeks.org/python-decimal-to-binary-list-conversion/
            #int(i) for i in bin(test_num)[2:]
        
            toMod.unsqueeze_(-1)
            toMod = toMod.expand(-1, 1, self.length_to_expand,8)
            for x in range(8):
                #Format command from: https://stackoverflow.com/a/16926357
                toMod[:, :, x].apply_(lambda y: int(format(int(y), '#010b')[x+2]))


        afterMod = torch.flatten(self.cnn(toMod),start_dim=1).to(input.device)

        input[:, :, :self.length_to_expand] = self.fc(afterMod).unsqueeze(1)
        return input






class Conv1DClassifier(AttackTrainingClassification):
    def __init__(self,mode="Soft",numberOfFeatures=1504):
        super().__init__(mode,numberOfFeatures)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, self.convolutional_channels[0], 3,device=device),
            self.activation,
            nn.MaxPool1d(self.maxpooling[0]),
            nn.Dropout(int(Config.parameters["Dropout"][0])))
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.convolutional_channels[0], self.convolutional_channels[1], 3,device=device),
            self.activation,
            nn.MaxPool1d(self.maxpooling[1]),
            nn.Dropout(int(Config.parameters["Dropout"][0])))
        
        sequencePackage = nn.Sequential()
        
        sequencePackage.append(self.layer1)
        sequencePackage.append(self.layer2)
        if self.end.end_type!="DOC":
            if Config.dataparallel:
                sequencePackage.append(self.sequencePackage.module)
                self.sequencePackage = nn.DataParallel(sequencePackage)
            else:
                sequencePackage.append(self.sequencePackage)
                self.sequencePackage = sequencePackage

        
        

        



class FullyConnected(AttackTrainingClassification):
    def __init__(self,mode="Soft",numberOfFeatures=1504):
        super().__init__(mode,numberOfFeatures)
        self.layer1 = nn.Sequential(
            #Sorry about the big block of math, trying to calculate how big the convolutional tensor is after the first layer
            nn.Linear(numberOfFeatures,int(self.convolutional_channels[0]*(((numberOfFeatures)/(self.maxpooling[0])//1)-1)),device=device),
            self.activation,
            nn.Dropout(int(Config.parameters["Dropout"][0])))
        self.layer2 = nn.Sequential(
            nn.Linear(int(self.convolutional_channels[0]*(((numberOfFeatures)/(self.maxpooling[0])//1)-1)),self.fullyConnectedStart,device=device),
            self.activation,
            nn.Dropout(int(Config.parameters["Dropout"][0])))

        sequencePackage = nn.Sequential()
        
        sequencePackage.append(self.layer1)
        sequencePackage.append(self.layer2)
        if self.end.end_type!="DOC":
            if Config.dataparallel:
                sequencePackage.append(self.sequencePackage.module)
                self.sequencePackage = nn.DataParallel(sequencePackage)
            else:
                sequencePackage.append(self.sequencePackage)
                self.sequencePackage = sequencePackage



