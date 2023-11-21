import os, sys
from parser import pcap2df
test = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/ML_Model")
import ModelStruct
import Config


def load_model(dataframe):
	model = ModelStruct.Conv1DClassifier()
	ModelStruct.train_model(model)
	data_object = model.generateDataObject(dataframe)
	return data_object


if __name__ == "__main__":
	df = pcap2df("../../samplePackets.pcapng")
	model_output = load_model(df)
	len(model_output.unknowns) > 0
	assert len(model_output.attacks) == Config.parameters["CLASSES"][0] - 1
	assert model_output.num_packets > 0
	assert isinstance(model_output.unknowns, list)

