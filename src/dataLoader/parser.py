import pyshark
import os, sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
test = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "src"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/ML_Model")
import ModelStruct
import Config

other_protocols = ["OSPF", "SCTP", "GRE", "SWIPE", "MOBILE", "SUN-ND", "SEP", "UNAS", "PIM", "SECURE-VMTP", "PIPE", "ETHERIP", "IB", "AX.25", "IPIP", "SPS", "IPLT", "HMP", "GGP", "IPV6", "RDP", "RSVP", "SCCOPMCE", "EGP", "VMTP", "SNP", "CRTP", "EMCON", "NVP", "FIRE", "CRUDP", "GMTP", "DGP", "MICP", "LEAF-2", "ARP", "FC", "ICMP"]


def make_df(data):
	col_names = ["ttl", "total_len", "protocol", "t_delta"]
	cols = [f"PayloadByte{x+1}" for x in range(1500)] + col_names
	df = pd.DataFrame(data, columns=cols)
	return df


def pcap2df(in_file):
	data_array = [[]]
	pcap = pyshark.FileCapture(f'{in_file}')
	raw_pcap = pyshark.FileCapture(f'{in_file}', use_json=True, include_raw=True)
	for packet, raw_packet in zip(pcap, raw_pcap):
		raw = raw_packet.get_raw_packet()
		length = len(raw)
		protocol = 0
		t_delta = 0
		ttl = 0
		if "IP" in packet:
			ttl = packet.ip.ttl
			if "UDP" in packet:
				protocol = "udp"
				t_delta = packet.udp.time_delta
			elif "TCP" in packet:
				protocol = "tcp"
				t_delta = packet.tcp.time_delta
		if protocol == 0:
			for prot in other_protocols:
				if prot in packet:
					protocol = prot.lower()
			if protocol == 0:
				protocol = "other"
		data_array.append([byte for byte in raw] + [0 for _ in range(1500 - len(raw))] + [ttl, length, protocol, t_delta])
	df = make_df(data_array)
	model = ModelStruct.Conv1DClassifier()
	ModelStruct.train_model(model)
	model.generateDataObject(df)


if __name__ == "__main__":
	data_object = pcap2df("samplePackets.pcapng")
	len(data_object.unknowns) > 0
	assert len(data_object.attacks) == Config.parameters["CLASSES"][0] - 1
	assert data_object.num_packets > 0
	assert isinstance(data_object.unknowns, list)
	# assert len(batch[0]) == len(data_object.predictions)
	# assert len(batch[0]) == len(data_object.prediction_confidence)
