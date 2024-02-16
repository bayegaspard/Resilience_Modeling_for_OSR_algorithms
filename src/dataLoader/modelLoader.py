import os
from parser import pcap2df
os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import ML_Model

def load_model(dataframe):
	# TODO FileNotFoundError: [Errno 2] No such file or directory: 'datasets/UnitTestingcounts.csv'
	ML_Model.Config.parameters["dataset"][0] = "UnitTesting"
	model = ML_Model.ModelStruct.Conv1DClassifier()
	ML_Model.ModelStruct.train_model(model)
	data_object = model.generateDataObject(dataframe)
	return data_object


if __name__ == "__main__":
	df = pcap2df("./samplePackets.pcapng")
	model_output = load_model(df)
	assert len(model_output.attacks) == ML_Model.Config.parameters["CLASSES"][0] - 1
	assert model_output.num_packets > 0
	assert isinstance(model_output.unknowns, list)

