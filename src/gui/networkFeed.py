from clientDataLoader import ClientDataLoader
import pandas as pd
import pyshark
import Pyro5
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/dataLoader")
import parser as pcap

cols = [f"payload_byte_{x+1}" for x in range(1500)] + ["ttl", "total_len", "protocol", "t_delta"]
other_protocols = ["OSPF", "SCTP", "GRE", "SWIPE", "MOBILE", "SUN-ND", "SEP", "UNAS", "PIM", "SECURE-VMTP", "PIPE", "ETHERIP", "IB", "AX.25", "IPIP", "SPS", "IPLT", "HMP", "GGP", "IPV6", "RDP", "RSVP", "SCCOPMCE", "EGP", "VMTP", "SNP", "CRTP", "EMCON", "NVP", "FIRE", "CRUDP", "GMTP", "DGP", "MICP", "LEAF-2", "ARP", "FC", "ICMP"]


def feedSamplePackets():
    print("Preparing to read the dataset")
    df = pcap.pcap2df("../samplePackets.pcapng")
    print(df)
    client = ClientDataLoader()
    serializabledf = df.to_dict()
    print("Serialized the dataset")
    client.sendPackets(serializabledf)


# def parsePacket(packet, raw_packet):
#     raw = raw_packet.get_raw_packet()
#     length = len(raw)
#     if length > 1500:
#         # print(packet)
#         return None
#     protocol = 0
#     t_delta = 0
#     ttl = 0
#     if "IP" in packet:
#         if packet.ip.src == "":
#             print("No source IP address")
#         if packet.ip.dst == "":
#             print("No destination IP address")
#         ttl = float(packet.ip.ttl)
#         if "UDP" in packet:
#             # if packet.udp.port == PYRO_PORT:
#             #   return None  # skip pyro packets
#             protocol = "udp"
#             t_delta = float(packet.udp.time_delta)
#         elif "TCP" in packet:
#             protocol = "tcp"
#             t_delta = float(packet.tcp.time_delta)
#     else:
#         print("No IP layer")
#     if protocol == 0:
#         for prot in other_protocols:
#             if prot in packet:
#                 protocol = prot.lower()
#         if protocol == 0:
#             protocol = "other"
#     return [byte for byte in raw] + [0 for _ in range(1500 - len(raw))] + [ttl, length, protocol, t_delta]


def feedNetwork(interface):
    if interface is None:
        interface = any
        print("You should supply an interface for listening to the network. Because you didnt, all interfaces are being sniffed")
    if str(interface).startswith("w") or interface is None:
        print("Please stop using WiFi; packets will be discarded. Use Ethernet. Thanks")

    feed = pyshark.LiveCapture(interface="\\Device\\NPF_{5A8EEC35-5F07-425C-A9D5-F087D02A8E6D}", use_json=True, include_raw=True)

    batch = []
    for packet in feed.sniff_continuously():
        parsedPacket = pcap.parsePacket(raw_packet=packet)
        if parsedPacket is not None:
            batch.append(parsedPacket)

        if len(batch) > 99:
            try:
                df = pcap.make_df(batch)  # pd.DataFrame(batch, columns=cols)
                print(df)
                client = ClientDataLoader()
                serializabledf = df.to_dict()
                # print(serializabledf)
                print("Serialized the batch")
                client.sendPackets(serializabledf)
            except Exception:
                print("Pyro5 traceback: ")
                print("".join(Pyro5.errors.get_pyro_traceback()))
            batch = []


if __name__ == "__main__":
    # feedSamplePackets()
    feedNetwork(interface=None)
