from nids_transformers import PADEC

global padec
global initialized
initialized = False;

if initialized is False:
    padec = PADEC()
    print(initialized)

def init():
    global padec
    global initialized

    print(padec)
    initialized = True
    print("initialized!")


def generate(header, payload):
    print(padec)
    packet_hex = header + payload
    tags = padec.GenerateTags(packet_hex_stream=packet_hex,
                              forward_packets_per_second=0,
                              backward_packets_per_second=0,
                              bytes_transferred_per_second=0,
                              total_tags=10)

    print(tags)