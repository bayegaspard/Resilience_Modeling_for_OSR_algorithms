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
            self.data_loader = Pyro5.api.Proxy(uriPyro)
            self.data_loader.setMac(get_mac())
            self.data_loader.connectModel()
        except Pyro5.errors.ConnectionClosedError:
            print("The connection to the server could not be established")

    def getPackets(self, timerange, category):
        if self.data_loader is None:
            print("An attempt was made to retrieve packets, but there is no connection to the server")
            return None
        results = self.data_loader.getPackets(timerange, category)
        print(f"Retrieved {len(results)} packet(s)")
        return results

    def sendPackets(self, payload):
        if self.data_loader is None:
            print("An attempt was made to send packets, but there is no connection to the server")
            return None
        self.data_loader.uploadPackets(payload)
