#from clientDataLoader import ClientDataLoader
import pandas as pd
import nest_asyncio
import pyshark
import Pyro5
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/dataLoader")
import parser as pcap
from cfg import config_interface

cfg = config_interface()
cols = [f"payload_byte_{x+1}" for x in range(1500)] + ["ttl", "total_len", "protocol", "t_delta"]
other_protocols = ["OSPF", "SCTP", "GRE", "SWIPE", "MOBILE", "SUN-ND", "SEP", "UNAS", "PIM", "SECURE-VMTP", "PIPE", "ETHERIP", "IB", "AX.25", "IPIP", "SPS", "IPLT", "HMP", "GGP", "IPV6", "RDP", "RSVP", "SCCOPMCE", "EGP", "VMTP", "SNP", "CRTP", "EMCON", "NVP", "FIRE", "CRUDP", "GMTP", "DGP", "MICP", "LEAF-2", "ARP", "FC", "ICMP"]

nest_asyncio.apply()


def feedSamplePackets(loader):
    print("Preparing to read the dataset")
    df = pcap.pcap2df("../samplePackets.pcapng")
    serializabledf = df.to_dict()
    print("Serialized the dataset")
    loader.sendPackets(serializabledf)


def feedNetworkThr(interface=any, loader=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop=loop)
    loop.run_until_complete(feedNetwork(interface=interface, loader=loader))
    loop.close()


async def feedNetwork(interface=any, loader=None):
    if interface is any:
        print("You should supply an interface for listening to the network. Because you didnt, all interfaces are being sniffed")
    if str(interface).startswith("w") or interface is None:
        print("Please stop using WiFi; packets will be discarded. Use Ethernet. Thanks")
    print("Intialized feedNetwork")

    pcapFilePath = "src/pcap/pcapFile.pcap"
    if os.path.exists(pcapFilePath):
        os.remove(pcapFilePath)

    feed = pyshark.LiveCapture(interface=cfg("interface"), use_json=True, include_raw=True, output_file=pcapFilePath)

    batch = []
    for packet in feed.sniff_continuously():
        parsedPacket = pcap.parsePacket(raw_packet=packet)
        if parsedPacket is not None:
            batch.append(parsedPacket)

        if len(batch) > 99:
            try:
                df = pcap.make_df(batch)
                print(f"Sending a batch of {len(batch)} packets.")
                loader.newPackets(df)
            except Exception:
                print("Pyro5 traceback: ")
                print("".join(Pyro5.errors.get_pyro_traceback()))
            batch = []
    await feed.close_async()


if __name__ == "__main__":
    # feedSamplePackets()
    feedNetwork(interface=None)
