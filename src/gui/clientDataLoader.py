import os
import sys
import Pyro5.api
from uuid import getnode as get_mac
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


uriPyro = "PYRO:dataloader@localhost:58116"  # move to config file somewhere


class ClientDataLoader:
    def __init__(self):
        print("ClientDataLoader has been initialized")
        try:
            # Pyro5.config.SERIALIZER = "marshal"
            self.data_loader = Pyro5.api.Proxy(uriPyro)
            self.data_loader.setMac(get_mac())
            self.data_loader.connectModel()
        except Pyro5.errors.ConnectionClosedError:
            print("The connection to the server could not be established")
        except Pyro5.errors.CommunicationError:
            print("Communcations were not successful with the server")

    def getClassBins(self, timerange, binct, filters):
        if self.data_loader is None:
            print("An attempt was made to retrieve the bins, but there is no connection to the server")
            return None
        try:
            results = self.data_loader.getClassBins(timerange, binct, filters)
            return results
        except Exception as e:
            print(f"getPackets exception: {e}")
            print("".join(Pyro5.errors.get_pyro_traceback()))
            return None

    def getClassCounts(self, timerange, filters):
        if self.data_loader is None:
            print("An attempt was made to retrieve a packet, but there is no connection to the server")
            return None
        try:
            results = self.data_loader.getClassCounts(timerange, filters)
            return results
        except Exception as e:
            print(f"getPackets exception: {e}")
            print("".join(Pyro5.errors.get_pyro_traceback()))
            return None

    def getPacket(self, id):
        if self.data_loader is None:
            print("An attempt was made to retrieve a packet, but there is no connection to the server")
            return None
        try:
            result = self.data_loader.getPacket(id)
            return result
        except Exception as e:
            print(f"getPackets exception: {e}")
            print("".join(Pyro5.errors.get_pyro_traceback()))
            return None

    def getPackets(self, timerange, category, requestData):
        if self.data_loader is None:
            print("An attempt was made to retrieve packets, but there is no connection to the server")
            return None
        try:
            results = {'rowData': self.data_loader.getPackets(timerange, category, requestData)}
            if len(results['rowData']) > 0:
                results['rowCount'] = results["rowData"][0]["full_count"]
            else:
                results['rowCount'] = 0
            print(results["rowCount"])
            # print(results)
            # print(f"Retrieved {len(results['rowData'])} packet(s)")
            return results
        except Exception as e:
            print(f"getPackets exception: {e}")
            print("".join(Pyro5.errors.get_pyro_traceback()))
            return None

    def sendPackets(self, payload):
        if self.data_loader is None:
            print("An attempt was made to send packets, but there is no connection to the server")
            return None
        self.data_loader.uploadPackets(payload)

    def listModels(self):
        if self.data_loader is None:
            print("An attempt was made to retrieve a list of packets, but there is no connection to the server")
            return None
        return self.data_loader.listModels()

    def loadModel(self, save):
        if self.data_loader is None:
            print("An attempt was made to load a model, but there is no connection to the server")
            return None
        return self.data_loader.loadModel(save)
