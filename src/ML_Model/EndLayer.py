import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import Config
import helperFunctions

# three lines from https: //xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
root_folder = os.path.abspath(os.path.dirname(root_folder))
sys.path.append(root_folder)

root_path = os.getcwd()


class EndLayers(nn.Module):

    def __init__(self, num_classes: int, cutoff=0.25, type="Soft"):
        """
        Endlayers is a module that takes the final layer of the direct neural network and appplies one of several functions to it to separate unknowns.

        parameters:
            num_classes - the number of classes to sort into.
            cutoff - what value to use to cut things off and declare as unknowns. Not applied to Softmax (that would be softmax threshold.).
            type - what function to use.
        """
        super().__init__()
        self.cutoff = cutoff
        self.classCount = num_classes
        self.end_type = type
        self.DOO = Config.parameters["Degree of Overcompleteness"][0]    # Degree of Overcompleteness for COOL
        self.weibulInfo = None
        self.var_cutoff = Config.parameters["Var_filtering_threshold"][0]
        self.resetvals()

    def forward(self, output_true: torch.Tensor, y=None, type=None) -> torch.Tensor:
        """
        Module forward command, is used by __call__().
        This function takes the logits from the privious models and evaluates them according to the selected function.

        Note: this function requires self.prepWeibull() to have been called earlier. That stores some external data that some functions require.

        parameters:
            output_true - The true base level output logits from the last fully connected layer.
            y - The true labels of each item. Used to generate a precision recall curve score.
            type - Type override for what function to use.
        """
        # check if a type is specified
        if type is None:
            type = self.end_type

        if y is not None and False:
            if type == "Energy":
                # Energy kind of reverses things.
                self.rocData[0] = y == Config.parameters["CLASSES"][0]
            else:
                self.rocData[0] = y != Config.parameters["CLASSES"][0]

        # modify outputs if nessisary for algorithm
        output_modified = self.typesOfMod.get(type, self.typesOfMod["none"])(self, output_true)

        # This is supposted to add an extra column for unknowns
        output_complete = self.typesOfUnknown[type](self, output_modified)

        if self.var_cutoff > 0 and type not in ["COOL"]:
            output_m_soft = self.typesOfMod.get("Soft", self.typesOfMod["none"])(self, output_true)
            output_c_soft = self.typesOfUnknown["Soft"](self, output_m_soft)
            var_mask = ~self.varmax_mask(output_true)
            output_complete[var_mask] = output_c_soft[var_mask]

        return output_complete

    # setup
    def setArgs(self, classes=None, weibullThreshold=0.9, weibullTail=20, weibullAlpha=3, score="energy", m_in=-1, m_out=0, temp=None):
        """
        Internal function for setting arguements that are used for some functions.
        This function creates a class Args that stores the values so that they can be called with args.value
        We have done this to preserve some of the original code for implementations.
        """
        param = pd.read_csv(os.path.join("Saves", "hyperparam", "hyperParam.csv"))
        unknowns = pd.read_csv(os.path.join("Saves", "unknown", "unknowns.csv"))
        unknowns = unknowns["Unknowns"].to_list()
        if temp is None:
            temp = float(param["Temperature"][0])
        if classes is None:
            classes = len(Config.parameters["Knowns_clss"][0])

        class argsc():
            def __init__(self):
                """
                This class is purely for formatting because the original implementation used an arguement parser
                """
                # OpenMax
                self.train_class_num = classes
                self.weibull_threshold = weibullThreshold
                self.weibull_tail = weibullTail
                self.weibull_alpha = weibullAlpha
                # EnergyOOD
                self.score = score
                self.m_in = m_in
                self.m_out = m_out
                self.T = temp
        args = argsc()

        self.args = args

    def prepWeibull(self, trainloader, device, net):
        """
        This stores a trainloader and training net for future refrence.
        Some of the algorithms want to create some sort of average values over the train loader. So this is to facilitate that.
        """
        self.weibulInfo = {"loader": trainloader, "device": device, "net": net, "weibull": None}

    def noChange(self, X: torch.Tensor):
        """
        A nothing function if no function is to be applied.
        """
        return X

    # ---------------------------------------------------------------------------------------------
    # This is the section for adding unknown column

    def softMaxUnknown(self, percentages: torch.Tensor):
        """
        This is just Softmax. It adds a column of zeros to fit in with the rest of the algorithms.
        """
        # self.Save_score.append(percentages.max(dim=1)[0].mean())
        # this adds a row that is of the cutoff amout so unless there is another value greater it will be unknown
        batchsize = len(percentages)
        unknownColumn = torch.zeros(batchsize, device=percentages.device)
        # self.Save_score.append(unknownColumn)
        # self.rocData[1] = unknownColumn
        return torch.cat((percentages, unknownColumn.unsqueeze(1)), dim=1)

    def normalThesholdUnknown(self, percentages: torch.Tensor):
        """
        This is a verson of softmax with threshold. It was turned down because "we cannot make changes to Softmax"
        """
        # self.rocData[1] = percentages.max(dim=1)[0]
        # self.Save_score.append(self.rocData[1].mean())
        # this adds a row that is of the cutoff amout so unless there is another value greater it will be unknown
        batchsize = len(percentages)
        unknownColumn = self.cutoff * torch.ones(batchsize, device=percentages.device)
        return torch.cat((percentages, unknownColumn.unsqueeze(1)), dim=1)

    def energyUnknown(self, percentages: torch.Tensor):
        """
        Modifies the output logits with energy based distribution and adds a column for unknowns.
        """
        if self.args is None:
            self.setArgs()
        import CodeFromImplementations.EnergyCodeByWetliu as Eng  # useful link to import in relative directories https: //stackoverflow.com/questions/4383571/importing-files-from-different-folder

        # The energy score code likes to output as a list
        scores = []
        Eng.energyScoreCalc(scores, percentages, self.args)
        scores = torch.tensor(np.array(scores), device=percentages.device)
        # after converting it to a tensor, the wrong dimention is expanded
        scores = -scores.squeeze(dim=0).unsqueeze(dim=1)
        # This was to print the scores it was going to save
        # print(scores.sum()/len(scores))
        # Just store this for later
        # self.rocData[1] = -scores
        # self.Save_score.append(scores.mean())
        # once the dimentions are how we want them we test if it is above the cutoff
        scores = scores.less_equal(self.cutoff).to(torch.int)
        # Then we run precentages through a softmax to get a nice score
        percentages = torch.softmax(percentages, dim=1)
        # Finally we join the results as an unknown class in the output vector
        return torch.cat((percentages, scores), dim=1)

    def DOCUnknown(self, percentages: torch.Tensor):
        """
        I do not understand how DOC works but this runs the code we were able to find and returns a one hot prediction metrix.
        """
        import CodeFromImplementations.DeepOpenClassificationByLeishu02 as DOC
        if self.docMu is None:
            # print("Mu Standards need to be collected")
            if self.weibulInfo is None:
                return
            else:
                self.docMu = DOC.muStandardsFromDataloader(Config.parameters["Knowns_clss"][0], self.weibulInfo["loader"], self.weibulInfo["net"])
                # self.Save_score = [torch.tensor(self.docMu)[:, 1]]

        # self.rocData[1] = []
        newPredictions = DOC.runDOC(percentages.detach().cpu().numpy(), self.docMu, Config.parameters["Knowns_clss"][0], self.rocData[1])
        newPredictions = torch.tensor(newPredictions)
        for x in range(len(newPredictions)):
            newPredictions[x] = torch.tensor(helperFunctions.rerelabel[newPredictions[x].item()])

        # to fit with the rest of the endlayers I am setting this back to a one hot vector even though we are just going to colapse it again.
        oneHotPredictions = F.one_hot(newPredictions, num_classes=Config.parameters["CLASSES"][0] + 1).float()

        return oneHotPredictions

    def iiUnknown(self, percentages):
        """
        Uses the intra-spread and inter-separation to classify items as unknowns or knowns.
        This score is then checked against the cutoff value and if it is less than the cutoff the unknown column is set to 2.
        The rest of the columns are limited in the range (0, 1) so argmax will always evaluate to the unknown if the cutoff is not reached.
        """
        import CodeFromImplementations.iiMod as iiMod
        unknowns = []
        for i in range(len(percentages)):
            unknowns.append(iiMod.outlier_score(percentages[i], self.iiLoss_means))
        unknowns = torch.stack(unknowns)
        percentages = iiMod.iimod(percentages, self.iiLoss_means).softmax(dim=1)

        # self.rocData[1] = unknowns  # I do not know if this is correct
        unknowns = 2 * unknowns.less_equal(self.cutoff)

        return torch.cat([percentages, unknowns.unsqueeze(dim=-1)], dim=-1)

    def varmax_final(self, logits: torch.Tensor):
        self.rocData[1] = torch.var(torch.abs(logits), dim=1)
        var_mask = self.varmax_mask(logits)
        shape = logits.shape
        unknown = torch.zeros([shape[0], 1], device=logits.device)
        unknown[var_mask] = 2
        output = torch.concat([torch.softmax(logits, dim=-1), unknown], dim=-1)
        return output

    def varmax_mask(self, logits):
        return torch.var(torch.abs(logits), dim=1) < self.var_cutoff

    # all functions here return a mask with 1 in all valid locations and 0 in all invalid locations
    typesOfUnknown = {"Soft": softMaxUnknown, "Energy": energyUnknown, "COOL": normalThesholdUnknown, "SoftThresh": normalThesholdUnknown, "DOC": DOCUnknown, "iiMod": iiUnknown, "Var": varmax_final}

    # ---------------------------------------------------------------------------------------------
    # This is the section for modifying the outputs for the final layer

    def softMaxMod(self, percentages: torch.Tensor):
        """
        Just runs softmax
        """
        return torch.softmax(percentages, dim=1)

    def odinMod(self, percentages: torch.Tensor):
        """
        Some prerequisites for ODIN. ODIN does not currently work.
        """
        print("ODIN is not working at the moment")
        import CodeFromImplementations.OdinCodeByWetliu as Odin
        self.model.openMax = False
        new_percentages = torch.tensor(Odin.ODIN(self.OdinX, self.model(self.OdinX), self.model, self.temp, self.noise))
        self.model.openMax = True
        return new_percentages[: len(percentages)]

    def FittedLearningEval(self, percentages: torch.Tensor):
        """
        Collapses COOL into the standard number of classes from the increased compettitive number of classes.
        After that it is just standard Softmax Unknown.
        """
        import CodeFromImplementations.FittedLearningByYhenon as fitted
        per = percentages.softmax(dim=1)
        store = []
        for x in per:
            store.append(fitted.infer(x, self.DOO, self.classCount))
        store = np.array(store)
        return torch.tensor(store)

    def DOCmod(self, logits: torch.Tensor):
        """
        DOC uses a sigmoid layer.
        """
        percent = torch.sigmoid(helperFunctions.renameClasses(logits))
        return percent

    def iiLoss_Means(self, percentages: torch.Tensor):
        """
        This just calculates and saves the means for each class for use in iiUnknown()
        """
        import CodeFromImplementations.iiMod as iiMod
        self.iiLoss_means = iiMod.Algorithm_1(self.weibulInfo["loader"], self.weibulInfo["net"])
        return percentages

    # all functions here return a tensor, sometimes it has an extra column for unknowns
    typesOfMod = {"Soft": softMaxMod, "Odin": odinMod, "COOL": FittedLearningEval, "SoftThresh": softMaxMod, "DOC": DOCmod, "iiMod": iiLoss_Means, "none": noChange, "Var": noChange}

    # ---------------------------------------------------------------------------------------------
    # This is the section for training label modification

    def FittedLearningLabel(self, labelList: torch.Tensor):
        """
        COOL changes the structure of the model to have a multiple of the number of classes.
        This modifies the labels for training so that cross entropy training works with this expanded end layer.
        """
        import CodeFromImplementations.FittedLearningByYhenon as fitted
        store = []
        for x in labelList:
            store.append(fitted.build_label(x, self.classCount, self.DOO))
        store = np.array(store)
        return torch.tensor(store, device=labelList.device)

    typesOfLabelMod = {"COOL": FittedLearningLabel}

    def labelMod(self, labelList: torch.Tensor):
        """
        A way of creating a dictionary with a default value. There might be a simpler way of doing this.
        """
        if self.end_type in self.typesOfLabelMod.keys():
            return self.typesOfLabelMod.get(self.end_type)(self, labelList)
        else:
            return self.noChange(labelList)

    # ---------------------------------------------------------------------------------------------
    # Some have specific training methods

    def iiTrain(self, batch, model):
        """
        iimod adds a training function that needs to be used in order to group the output logits.
        (Energy also adds an optional training function as well but we did not add it (iimod was more of a last minute addition))
        """
        import CodeFromImplementations.iiMod as iiMod
        iiMod.singleBatch(batch, model)

    typesOfTrainMod = {"iiMod": iiTrain}

    def trainMod(self, batch, model):
        """
        A way of creating a dictionary with a default value. There might be a simpler way of doing this.
        """
        if self.end_type in self.typesOfTrainMod.keys():
            return self.typesOfTrainMod.get(self.end_type)(self, batch, model)
        else:
            return

    # ---------------------------------------------------------------------------------------------
    # This is the section for resetting each epoch

    def resetvals(self):
        """
        Resets the values in the Endlayer object.
        """
        self.args = None    # This is the arguements for OPENMAX
        self.Save_score = []    # This is saving the score values for threshold for testing
        self.docMu = None    # This is saving the muStandards from DOC so that they dont need to be recalculated
        if (self.weibulInfo is not None) and (not self.weibulInfo["weibull"] is None):
            self.weibulInfo["weibull"] = None
        self.rocData = [[], []]  # This is the data for ROC of unknowns. First value is 1 if known data and 0 if unknown, second is the number before theshold.

    def distance_by_batch(self, labels: torch.Tensor, outputs: torch.Tensor, means: list):
        """
        Finds the distances using iiMod's intra_spread function.
        """
        from CodeFromImplementations.iiMod import intra_spread
        outputs_for_known_columns = outputs[:, Config.parameters["Knowns_clss"][0]]
        anti_unknown_value_mask = labels != Config.parameters["CLASSES"][0]
        return intra_spread(outputs_for_known_columns[anti_unknown_value_mask], means, labels[anti_unknown_value_mask])
