import pyshark
import os, sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

other_protocols = ["OSPF", "SCTP", "GRE", "SWIPE", "MOBILE", "SUN-ND", "SEP", "UNAS", "PIM", "SECURE-VMTP", "PIPE", "ETHERIP", "IB", "AX.25", "IPIP", "SPS", "IPLT", "HMP", "GGP", "IPV6", "RDP", "RSVP", "SCCOPMCE", "EGP", "VMTP", "SNP", "CRTP", "EMCON", "NVP", "FIRE", "CRUDP", "GMTP", "DGP", "MICP", "LEAF-2", "ARP", "FC", "ICMP"]


def make_df(data):
	col_names = ["ttl", "total_len", "protocol", "t_delta", "src", "dest", "time"]
	cols = [f"payload_byte_{x+1}" for x in range(1500)] + col_names
	df = pd.DataFrame(data, columns=cols)
	return df


def parsePacket(raw_packet):
	packet = raw_packet
	# guaranteed fields
	raw = raw_packet.get_raw_packet()
	# packet.length always == packet.captured_length. why are both of these a thing?
	length = packet.length
	if len(raw) != length:
		print(f"We've got two different packets?: {len(raw) - int(length)}")
	if int(length) > 1500 or len(raw) > 1500:
		print("packet too long")
		return None
	print(int(length))
	time = packet.sniff_timestamp
	t_delta = float(packet.frame_info.time_delta)
	# will, at least, be "other"
	protocol = ""
	# may stay empty
	ttl = 0
	src = ""
	dest = ""
	if "IP" in packet:
		ttl = int(packet.ip.ttl)
		src = str(packet.ip.src)
		dest = str(packet.ip.dst)
		if "UDP" in packet:
			protocol = "udp"
		elif "TCP" in packet:
			protocol = "tcp"
	elif "ARP" in packet:
		src = str(packet.arp.src.proto_ipv4)
		dest = str(packet.arp.dst.proto_ipv4)
		protocol = "arp"
	elif "IPV6" in packet:
		src = str(packet.ipv6.src)
		dest = str(packet.ipv6.dst)
		ttl = int(packet.ipv6.hlim)
		protocol = "ipv6"
	if protocol == 0:
		for prot in other_protocols:
			if prot in packet:
				protocol = prot.lower()
		if protocol == 0:
			protocol = "other"
	return [byte for byte in raw] + [0 for _ in range(1500 - len(raw))] + [ttl, len(raw), protocol, t_delta, src, dest, time]


# TODO: use with parsePacket instead
def pcap2df(in_file):
	data_array = [[]]
	pcap = pyshark.FileCapture(f'{in_file}')
	raw_pcap = pyshark.FileCapture(f'{in_file}', use_json=True, include_raw=True)
	for packet, raw_packet in zip(pcap, raw_pcap):
		# guaranteed fields
		raw = raw_packet.get_raw_packet()
		# packet.length always == packet.captured_length. why are both of these a thing?
		length = packet.length
		time = packet.sniff_timestamp
		t_delta = packet.frame_info.time_delta
		# will, at least, be "other"
		protocol = ""
		# may stay empty
		ttl = 0
		src = ""
		dest = ""
		if "IP" in packet:
			ttl = packet.ip.ttl
			src = packet.ip.src
			dest = packet.ip.dst
			if "UDP" in packet:
				protocol = "udp"
			elif "TCP" in packet:
				protocol = "tcp"
		elif "ARP" in packet:
			src = packet.arp.src_proto_ipv4
			dest = packet.arp.dst_proto_ipv4
			protocol = "arp"
		elif "IPV6" in packet:
			src = packet.ipv6.src
			dest = packet.ipv6.dst
			ttl = packet.ipv6.hlim
			protocol = "ipv6"
		if protocol == 0:
			for prot in other_protocols:
				if prot in packet:
					protocol = prot.lower()
			if protocol == 0:
				protocol = "other"
		data_array.append([byte for byte in raw] + [0 for _ in range(1500 - len(raw))] + [ttl, length, protocol, t_delta, src, dest, time])
	return make_df(data_array)

if __name__ == "__main__":
	pcap2df("samplePackets.pcapng")