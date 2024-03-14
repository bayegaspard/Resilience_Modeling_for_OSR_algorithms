import re
import subprocess

def get_hex_dumps_for_packets(packet_numbers):
    """
    Extracts the hex dumps for multiple packets based on a filter string using tshark.

    :param packet_numbers: a list of packet numbers
    :return: Filtered list of hex strings and their corresponding packet numbers, written like so: "{packet_number}:{hex_string}"
    """
    # Construct the tshark command with the filter
    file_path = "/home/ncostagliola/PycharmProjects/Open-World-Recognition-Tool/src/pcap/pcapFile.pcap"
    filter_str = " || ".join([f"frame.number == {num}" for num in packet_numbers])

    #This first command filters the packet_numbers list to get only the packets that PADEC will accept.
    #Hopefully not necessary in the future w/ new packet description model
    packet_numbers_cmd = f"tshark -r {file_path} -Y '((tcp || udp) && ({filter_str})) && (tcp.len > 0)' -T fields -e frame.number"
    process = subprocess.Popen(packet_numbers_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, _ = process.communicate()

    # Decode and split the output to get a list of packet numbers
    matching_packet_numbers = output.decode().strip().split('\n')
    filter_str = " || ".join([f"frame.number == {num}" for num in matching_packet_numbers ])

    #This command uses the filtered packet number list to fetch the hex dumps.
    tshark_cmd = f"tshark -r {file_path} -Y '((tcp || udp) && ({filter_str})) && (tcp.len > 0)' -x -s 0"
    # Execute the command
    process = subprocess.Popen(tshark_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    packets = output.decode().split('\n\n')

    if process.returncode == 0:
        return hex_dump_to_strings(packets, matching_packet_numbers)
    else:
        print(f"Error getting hex dumps: {err.decode('utf-8')}")
        return None


def hex_dump_to_strings(packets, packet_numbers):
    """
    Converts hex dumps to strings

    :param packet_numbers: a list of packet numbers
    :param packets: a list of hex dumps
    :return: Filtered list of hex strings and their corresponding packet numbers, written like so: "{packet_number}:{hex_string}"
    """

    packet_strings_with_numbers = []

    for packet_number, packet in zip(packet_numbers, packets):

        lines = packet.split('\n')
        hex_string = ''
        for line in lines:
            # Extract the hex data from columns 6 to 53. This is to ignore parts of the hex dump we don't want.
            hex_data = line[6:53]
            # Extract only the hex characters and spaces, then remove spaces. Append each line to the hex string.
            matches = re.findall(r'[0-9a-f]{2}', hex_data, re.IGNORECASE)
            hex_string += ''.join(matches)

        #Attach corresponding packet number
        packet_string_with_number = f"{packet_number}:{hex_string}"
        packet_strings_with_numbers.append(packet_string_with_number)

    return packet_strings_with_numbers