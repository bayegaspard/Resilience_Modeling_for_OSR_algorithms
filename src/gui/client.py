import gui
from clientDataLoader import ClientDataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/dataLoader")
import parser as pcap


class Client(object):
    def __init__(self):
        df = pcap.pcap2df("../samplePackets.pcapng")
        print(df)
        self.startGUI()

    def feedSamplePackets(self):
        client = ClientDataLoader()

    def startGUI(self):
        self.gui = gui.GUI()
        self.gui.run()


if __name__ == "__main__":
    client = Client()
