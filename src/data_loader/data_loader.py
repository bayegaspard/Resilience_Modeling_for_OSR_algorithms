import Pyro5.api
import pandas as pd

# read permitted mac addresses


class DataLoader(object):
    def __init__(self):
        print("Initialized DataLoader")

    def testPrint(self):
        print("Test print")

    def loadModel(self, id, path):
        print("Loading model")


loader = DataLoader()


@Pyro5.api.expose
class DataLoaderInterface(object):
    def testPrint(self):
        loader.testPrint()

    def getPackets(self, timerange, category):
        print("Getting packets")

        return pd.read_csv("../sample.csv").to_dict(orient="rows")


daemon = Pyro5.api.Daemon()
uri = daemon.register(DataLoaderInterface)

print(uri)
daemon.requestLoop()
