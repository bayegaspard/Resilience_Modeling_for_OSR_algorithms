import pyshark
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

pcap = pyshark.FileCapture('../samplePack.pcap')
for packet in pcap:
	if "IP" in packet:
		ip_len = packet.ip.len
		ttl = packet.ip.ttl
		print(ip_len)
		protocol = None
		if "UDP" in packet:
			protocol = "UDP"
			t_delta = packet.udp.time_delta
			print(packet.udp.field_names)
			udp_len = packet.udp.length
		elif "TCP" in packet:
			protocol = "TCP"
			tcp_len = packet.tcp.len
			t_delta = packet.tcp.time_delta
