import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from outputDataObject import outputDataUpdateObject


class Client:
    def __init__(self):
        self.outputDataUpdates = []
        self.data = outputDataUpdateObject()
        self.poll()

    def poll(self):
        self.outputDataUpdates.append(outputDataUpdateObject.dummy)

    def calculateAggregate(self):
        self.data = outputDataUpdateObject()
        for updateObject in self.outputDataUpdates:
            self.data.num_packets += updateObject.num_packets
            self.data.unknowns.extend(updateObject.unknowns)
            self.data.attacks.update(updateObject.attacks)
            self.data.datashiftFactor += updateObject.datashiftFactor
        self.data.datashiftFactor /= len(self.outputDataUpdates)
