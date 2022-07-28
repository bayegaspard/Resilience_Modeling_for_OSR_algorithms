import numpy as np
from LoadPackets import NetworkDataset
import torch
import torch.utils.data
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import Evaluation
from ModelLoader import Network
import OpenMaxByMaXu
import os



#pick a device
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

torch.manual_seed(0)
CLASSES = 36
BATCH = 500
NAME = "OpenMax"
CUTOFF = 0.1

#I looked up how to make a dataset, more information in the LoadImages file
#images are from: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
data_total = NetworkDataset(["MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv","MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"])
unknown_data = NetworkDataset(["MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv"])

CLASSES = len(data_total.classes)

data_train, data_test = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])
testing = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH, shuffle=False)
training =  torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH, shuffle=True)

#this needs to be improved 
data_total.isOneHot = False
data_train2, _ = torch.utils.data.random_split(data_total, [len(data_total)-1000,1000])
training2 = torch.utils.data.DataLoader(dataset=data_train2, batch_size=BATCH, shuffle=True)

#load the unknown data
unknowns = torch.utils.data.DataLoader(dataset=unknown_data, batch_size=BATCH, shuffle=False)


model = Network(CLASSES).to(device)
#initialize the counters, op for open because open is a keyword
soft = Evaluation.correctValCounter(CLASSES, cutoff=CUTOFF)
op = Evaluation.correctValCounter(CLASSES, cutoff=CUTOFF)

if os.path.exists(NAME+"/checkpoint.pth"):
    model.load_state_dict(torch.load(NAME+"/checkpoint.pth",map_location=device))
    epochs = 5
else:
    epochs = 5

epochs = 20
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)



for e in range(epochs+1):
    lost_amount = 0
    for batch, (X, y) in enumerate(training):
        X = X.to(device)
        y = y.to(device)

        _, output = model(X)
        lost_points = criterion(output, y)
        optimizer.zero_grad()
        lost_points.backward()


        optimizer.step()

        lost_amount += lost_points.item()

    

    if e>4:
        with torch.no_grad():
            model.eval()

            unknownscore = 0
            if e>8:
                #these three lines somehow setup for the openmax thing
                scores, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
                catagories = list(range(CLASSES))
                weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories,tailsize=10)

                op.setWeibull(weibullmodel)

            for batch,(X,y) in enumerate(testing):
                X = X.to(device)
                y = y.to("cpu")

                _, output = model(X)

                output = output.to("cpu")

                

                soft.evalN(output,y)
                if e>8:
                    op.evalN(output,y, type="Open")


            
            print(f"-----------------------------Epoc: {e}-----------------------------")
            print(f"Lost in training: {100*lost_amount/len(data_train)}")
            print("SoftMax:")
            soft.PrintEval()
            if e>8:
                print("OpenMax:")
                op.PrintEval()

            if e%5 == 4:
                torch.save(model.state_dict(), NAME+"/checkpoint.pth")

            soft.zero()
            op.zero()

            model.train()


#Everything past here is unknowns

with torch.no_grad():
    unknownscore = 0
    model.eval()
    #these three lines somehow setup for the openmax thing
    scores, mavs, distances = OpenMaxByMaXu.compute_train_score_and_mavs_and_dists(CLASSES,training2,device,model)
    catagories = list(range(CLASSES))
    weibullmodel = OpenMaxByMaXu.fit_weibull(mavs,distances,catagories,tailsize=10)

    op.setWeibull(weibullmodel)

    for batch,(X,y) in enumerate(unknowns):
        X = X.to(device)
        y = y.to("cpu")
        
        _, output = model(X)

        output = output.to("cpu")


        soft.evalN(output,y,offset=26)
        op.evalN(output,y,offset=26,type="Open")

    print("SoftMax:")
    soft.PrintUnknownEval()
    print("OpenMax:")
    op.PrintUnknownEval()

    soft.zero()
    op.zero()
    
    model.train()
