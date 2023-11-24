import os
import sys
import Pyro5.api
from uuid import getnode as get_mac
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


uriPyro = "PYRO:dataloader@localhost:58118"


class ClientDataLoader:
    def __init__(self):
        print("Init client")
        self.data_loader = Pyro5.api.Proxy(uriPyro)
        self.data_loader.setMac(get_mac())
        self.data_loader.connectModel(":shrug:")
        # print(self.data_loader.getMac())
        # self.outputDataUpdates = []

    def getPackets(self, timerange, category):
        # self.data_loader.testPrint()
        print("getting packets")
        results = self.data_loader.getPackets(timerange, category)
        # )
        # self.data_loader.getPackets(timerange, category))
        print(results)
        return results  # pd.read_csv("../sample.csv")
