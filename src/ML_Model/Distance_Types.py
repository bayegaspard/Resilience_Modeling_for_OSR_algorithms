import torch
import Config
import math


# Code from the iiMod file (Very Modified)
def distance_measures(Z: torch.Tensor, means: list, Y: torch.Tensor, distFunct) -> torch.Tensor:
    """
    Returns a distance created by distFunct between the means and the predicted outputs.
    Z is the output of the model that should be in the form of (I,C) where I is the number of items in the batch and C is the total number of classes.
    means is a list of means for each of the C classes.
    Y is the final predicted class label in the form of (I) where I is the number of items in the batch.

    Returns a zero dimentional Tensor
    """
    with torch.no_grad():
        intraspread = torch.tensor(0, dtype=torch.float32)
        N = len(Y)
        K = range(len(Config.get_global("knowns_clss")))
        # K = range(len([0,1,2]))
        # For each class in the knowns
        for j in K:
            if means[j].dim() != 0:
                # The mask will only select items of the correct class
                mask = (Y == Config.get_global("knowns_clss")[j]).cpu()
                # mask = Y==[0,1,2][j]

                # torch.flatten(x,start_dim=1,end_dim=-1)
                dist = abs(distFunct(means[j].cpu(), Z.cpu()[mask.cpu()]))
                if not math.isnan(dist):
                    intraspread += dist
        intraspread = intraspread / N
    return intraspread


# Equation 2 from iiMod file
def class_means(Z: torch.Tensor, Y: torch.Tensor):
    """
    Creates the class means from the final batch output and the true labels.
    Z is the output of the model that should be in the form of (I,C) where I is the number of items in the batch and C is the total number of classes.
    Y is the final true class label in the form of (I) where I is the number of items in the batch.
    Returns a list of X dementional tensors, one row for each class where X is also the number of classes.
    """
    means = [torch.tensor(0, requires_grad=False) for x in range(Config.get_global("CLASSES"))]
    # print(Y.bincount())
    for y in Config.get_global("knowns_clss"):
        # for y in [0,1,2]:
        # Technically only this part is actually equation 2 but it seems to want to output a value for each class.
        mask = (Y == y)
        Cj = mask.sum().item()
        sumofz = Z[mask].sum(dim=0)
        if Cj != 0:
            means[y] = sumofz / Cj
        else:
            means[y] = sumofz
    return means


def class_means_from_loader(weibulInfo):
    """
    Creates the class means from the information gathered for the weibul model using the final batch output and the true labels.
    inputs:
        weibulInfo - a dictonary containing:
                loader - a Dataloader containing the training data to be used to find the means
                net - the network that is being trained
    """
    # Note, this masking is only due to how we are handling model outputs.
    # If I was to design things again I would have designed the model outputs not to need this masking.
    data_loader = weibulInfo["loader"]
    model = weibulInfo["net"]

    classmeans = None
    for num, (X, Y) in enumerate(data_loader):
        # Getting the correct column (Nessisary for our label design)
        y = Y[:, 0]
        Z = model(X)    # Step 2
        if classmeans is None:
            classmeans = class_means(Z, y)
        elif len(Z) == Config.get_global("batch_size"):
            classmeans = [(x * num + y) / (num + 1) for x, y in zip(classmeans, class_means(Z, y))]
        else:
            means = class_means(Z, y)
            if means.dim > 0:
                classmeans = [(x * num * Config.get_global("batch_size") + y) / (num * Config.get_global("batch_size") + len(y)) for x, y in zip(classmeans, means)]
        del (X)
        del (Y)
        del (Z)
    return classmeans


# Derived from ChatGPT. Apparently.
def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")

    squared_distance = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    if squared_distance.dim() > 0:  # ADDED LINE
        distance = [math.sqrt(x) for x in squared_distance]  # ADDED LINE
    else:  # ADDED LINE
        distance = math.sqrt(squared_distance)

    return distance


class forwardHook():
    """
    This is a module that collects logits from a model at given points within the model.
    It is designed to be attached as a forward hook in either one or several places within the model.
    It is used by creating the forward hook, and then getting the final predicted values of the model and setting forwardHook.class_vals to those values (1 dimentional tensor)
    After the class values are set, it is assumed that the next run will be the assoicated values and so the means will be generated.
    """

    def __init__(self):
        """
        creates the forward hook, note: this does not set self.class_vals, that needs to be set manually with the predicted classes.
        """
        self.distances = {}
        self.class_vals = None  # these are the final classifications for each row in the batch
        self.means = {}
        self.distFunct = "intra_spread"
        self.rm = None

    def __call__(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
        """
        This is the function that activates the forward hook.
        The first pass generates the means that the future passes will use (note that self.class_vals needs to be set first).
        The future passes will generate some distance that will be used in a running total. To reset this running total use the reset() method.
        The running total is not returned. You can get it using the self.distances attribute.
        """
        # print("Forward hook called")
        with torch.no_grad():
            name = f"{module._get_name()}_{output[0].size()}"
            if self.class_vals is None:
                if output.ndim == 2:
                    self.class_vals = output.argmax(dim=1).cpu()
                else:
                    self.class_vals = output.cpu()
            else:
                if name not in self.means.keys():
                    self.means[name] = class_means(output, self.class_vals)
                if name not in self.distances.keys():
                    self.distances[name] = distance_measures(output, self.means[name], self.class_vals, dist_types_dict[self.distFunct])
                else:
                    self.distances[name] += distance_measures(output, self.means[name], self.class_vals, dist_types_dict[self.distFunct])

    def reset(self):
        self.distances = {x: 0 for x in self.distances.keys()}


dist_types_dict = {
    "Cosine_dist": lambda x1, x2: 1 - torch.nn.functional.cosine_similarity(x1, x2[:, :len(x1)]).sum(),
    "intra_spread": lambda x, y: (torch.linalg.norm(x - y[:, :len(x)], dim=0)**2).sum(),
    "Euclidean Distance": lambda x1, x2: torch.tensor([euclidean_distance(x1, y2) for y2 in x2[:, :len(x1)]]).sum()
}

# torch.nn.modules.module.register_module_forward_hook(forwardHook())
