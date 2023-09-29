# Just the idea for the data output object

class outputDataUpdateObject():
    def __init__(self):
        # Number of packets. Should be an integer
        self.num_packets = 0
        # Unknowns list. Contains sets of possible labels for the unknown packet
        self.unknowns = []
        # Attacks and data registry, Dictonary containing the number of each seen attack.
        self.attacks = {}
        # Datashift amount. Should be a percentage that is related to how similar the training data
        self.datashiftFactor = 1

    @staticmethod
    def dummy():
        outputObject = outputDataUpdateObject()
        outputObject.num_packets = 12
        outputObject.unknowns = [{"SSH", "Long_term", "High_severity"}, {"Portscan", "Low_severity"}]
        outputObject.attacks = {"Benign": 7, "SSH": 2, "Heartbleed": 1}
        return outputObject
