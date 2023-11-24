import pyshark
import os, sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import ML_Model

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
	return df

