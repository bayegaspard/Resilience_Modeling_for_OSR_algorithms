import re
import subprocess

def get_hex_dumps_for_packets(filter_str):
    """
    Extracts the hex dumps for multiple packets based on a filter string using tshark.

    :param filter_str: A `tshark` display filter string for the packets of interest.
    :return: Hex dumps string for the packets or None if an error occurred.
    """
    # Construct the tshark command with the filter
    file_path = "/home/ncostagliola/PycharmProjects/Open-World-Recognition-Tool/src/pcap/pcapFile.pcap"
    tshark_cmd = f"tshark -r {file_path} -Y '((tcp || udp) && ({filter_str})) && (tcp.len > 0 || udp.length > 8)' -x -s 0"
    # Execute the command
    process = subprocess.Popen(tshark_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()

    if process.returncode == 0:
        return hex_dump_to_strings(output.decode('utf-8'))
    else:
        print(f"Error getting hex dumps: {err.decode('utf-8')}")
        return None


def hex_dump_to_strings(hex_dump):
    # Use a regular expression to find all instances of two hexadecimal digits
    # Split the hex dump into separate packets assuming a blank line as separator
    print(hex_dump)
    packets = hex_dump.strip().split('\n\n')

    packet_strings = []
    for packet in packets:
        # Use a regular expression to find all instances of two hexadecimal digits
        matches = re.findall(r'\b[0-9a-f]{2}\b', packet, re.IGNORECASE)
        # Concatenate all matches into a single string
        hex_string = ''.join(matches)
        packet_strings.append(hex_string)

    return packet_strings