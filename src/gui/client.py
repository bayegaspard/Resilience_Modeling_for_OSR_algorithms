import os
import sys
import pandas as pd
import Pyro5.api
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


uriPyro = "PYRO:obj_fcf38f45cb5e4a8d9b93ac80697b4364@localhost:52672"


class Client:
    def __init__(self):
        self.data_loader = Pyro5.api.Proxy(uriPyro)
        self.outputDataUpdates = []

    def getPackets(self, timerange, category):
        self.data_loader.testPrint()
        return pd.DataFrame.from_records(self.data_loader.getPackets(timerange, category))  # pd.read_csv("../sample.csv")
