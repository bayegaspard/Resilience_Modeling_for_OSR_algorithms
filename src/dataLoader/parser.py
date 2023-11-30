import pyshark
import os, sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

other_protocols = ["OSPF", "SCTP", "GRE", "SWIPE", "MOBILE", "SUN-ND", "SEP", "UNAS", "PIM", "SECURE-VMTP", "PIPE", "ETHERIP", "IB", "AX.25", "IPIP", "SPS", "IPLT", "HMP", "GGP", "IPV6", "RDP", "RSVP", "SCCOPMCE", "EGP", "VMTP", "SNP", "CRTP", "EMCON", "NVP", "FIRE", "CRUDP", "GMTP", "DGP", "MICP", "LEAF-2", "ARP", "FC", "ICMP"]


def make_df(data):
	col_names = ["ttl", "total_len", "protocol", "t_delta", "src", "dest", "time", "srcport", "destport"]
	cols = [f"payload_byte_{x+1}" for x in range(1500)] + col_names
	df = pd.DataFrame(data, columns=cols)
	return df


def parsePacket(raw_packet):
	packet = raw_packet
	# guaranteed fields
	raw = raw_packet.get_raw_packet()
	# packet.length always == packet.captured_length. why are both of these a thing?
	length = packet.length
	if int(length) > 1500 or len(raw) > 1500:
		print(f"Packet too large: length of {len(raw)}")
		return None
	time = packet.sniff_timestamp
	t_delta = float(packet.frame_info.time_delta)
	# will, at least, be "other"
	protocol = ""
	# may stay empty
	ttl = 0
	src = ""
	dest = ""
	srcport = ""
	destport = ""
	if "IP" in packet:
		ttl = int(packet.ip.ttl)
		src = str(packet.ip.src)
		dest = str(packet.ip.dst)
		if "UDP" in packet:
			protocol = "udp"
			srcport = str(packet.udp.srcport)
			destport = str(packet.udp.dstport)
		elif "TCP" in packet:
			protocol = "tcp"
			srcport = str(packet.tcp.srcport)
			destport = str(packet.tcp.dstport)
	elif "ARP" in packet:
		src = str(packet.arp.src.proto_ipv4)
		dest = str(packet.arp.dst.proto_ipv4)
		protocol = "arp"
	elif "IPV6" in packet:
		src = str(packet.ipv6.src)
		dest = str(packet.ipv6.dst)
		ttl = int(packet.ipv6.hlim)
		protocol = "ipv6"
	if protocol == "":
		for prot in other_protocols:
			if prot in packet:
				protocol = prot.lower()
		if protocol == "":
			protocol = "other"
	return [byte for byte in raw] + [0 for _ in range(1500 - len(raw))] + [ttl, len(raw), protocol, t_delta, src, dest, time, srcport, destport]


def pcap2df(in_file):
	data_array = [[]]
	raw_pcap = pyshark.FileCapture(f'{in_file}', use_json=True, include_raw=True)
	for packet in raw_pcap:
		packet_row = parsePacket(packet)
		if packet_row is not None:
			data_array.append(packet_row)
	return make_df(data_array)


if __name__ == "__main__":
	pcap2df("samplePackets.pcapng")