import os
import time
import threading
from nids_transformers import PADEC
global padec

def initialize():
    print("Consumer initializing...")
    global padec
    padec = PADEC()
    print("Initialization complete.")

def signal_ready(signal_file_path):
    with open(signal_file_path, 'w') as f:
        f.write("Consumer ready")
    print("Consumer ready signal created.")

def wait_for_data(data_signal_path):
    print("Waiting for data...")
    while not os.path.exists(data_signal_path):
        time.sleep(1)
    print("Data signal detected.")


def identify_protocol(packet_hex):
    # Extract the IP header starting position: skip Ethernet header (14 bytes = 28 hex characters)
    ip_header_start = 28

    # The Protocol field is the 9th byte of the IP header
    # Since each byte is 2 hex characters, we add 16 hex characters to the start of the IP header
    protocol_pos = ip_header_start + 18  # 9 bytes into the IP header, but we start counting from 0

    # Extract the protocol field (2 hex characters)
    protocol_hex = packet_hex[protocol_pos:protocol_pos + 2]

    # Convert hex to int
    protocol = int(protocol_hex, 16)
    print(protocol_hex,protocol),

    # Identify the protocol
    if protocol == 6 or protocol == 17:
        return True
    else:
        return False

def process_data(data_file_path, lock):
    global padec
    header = "5e0e8bffb00b581122d37761080045000034000040004006ef4386580d1f8efb"
    print("Consumer starts processing data...")
    while True:
        with lock:
            with open(data_file_path, 'r') as file:
                lines = file.readlines()
            if not lines:  # If the file is empty, wait for more data
                time.sleep(1)
                continue

            data = lines.pop(0).strip()
            # Read the first line
            if data == "Stop":
                print("Stop signal received. Consumer is stopping.")
                break

            with open(data_file_path, 'w') as file:
                file.writelines(lines)

        try:
            packet_hex = data
            tags = padec.GenerateTags(packet_hex_stream=packet_hex,
                                      forward_packets_per_second=0,
                                      backward_packets_per_second=0,
                                      bytes_transferred_per_second=0,
                                      total_tags=10)
            print(tags)
        except Exception as e:
            if False:
                print("wow")

    # Process each string in the data list
def check_for_stop(stop_signal_path):
    return os.path.exists(stop_signal_path)

def cleanup(ready_signal_path, data_signal_path, data_file_path, stop_signal_path):
    if os.path.exists(ready_signal_path):
        os.remove(ready_signal_path)
        print("Ready signal file removed.")

    if os.path.exists(data_signal_path):
        os.remove(data_signal_path)
        print("Data signal file removed.")

    if os.path.exists(data_file_path):
        os.remove(data_file_path)
        print("Data file removed.")

    if os.path.exists(stop_signal_path):
        os.remove(stop_signal_path)
        print("Stop signal file removed.")

def start_consumer_threads(n, filepath):
    lock = threading.Lock()
    threads = []

    for i in range(n):
        thread = threading.Thread(target=process_data, args=(filepath, lock), name=f"Consumer-{i}")
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == '__main__':

    ready_signal_path = "/tmp/consumer_ready.signal"
    data_signal_path = "/tmp/data_available.signal"
    data_file_path = "/tmp/data.txt"
    stop_signal_path = "/tmp/stop.signal"
    cleanup(data_file_path, data_signal_path, ready_signal_path, stop_signal_path)

    initialize()
    signal_ready(ready_signal_path)
    wait_for_data(data_signal_path)
    start_consumer_threads(16, data_file_path)

    # remove the signal files to reset for the next cycle
    cleanup(data_file_path, data_signal_path, ready_signal_path, stop_signal_path)

