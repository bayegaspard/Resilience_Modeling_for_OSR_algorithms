import os
import time
import threading
import psycopg2
from src.cfg import config_interface
from nids_transformers import PADEC
import json


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

def process_data(data_file_path, lock):
    global padec
    cfg = config_interface()
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

        parts = data.split(':')#parts[0] is foreign key, parts[1] is the hex string

        try:
            tags = padec.GenerateTags(packet_hex_stream=parts[1],
                                      forward_packets_per_second=0,
                                      backward_packets_per_second=0,
                                      bytes_transferred_per_second=0,
                                      total_tags=5)
            print(tags)
        except Exception as e:
            continue

        try:
            #(cfg("DB_name"), cfg("DB_user"), cfg("DB_host"), cfg("DB_pw"), cfg("DB_port"))
            conn = psycopg2.connect(
                dbname=cfg("DB_name"),
                user=cfg("DB_user"),
                password=cfg("DB_pw"),
                host=cfg("DB_host"),
                port=cfg("DB_port")
            )
            c = conn.cursor()
            c.execute("INSERT INTO PACK_TAG (PACK_ID, TAGS) VALUES (%s, %s)",
                      (parts[0], json.dumps(tags)))
            conn.commit()
            c.close()
            conn.close()
            print("Tags inserted into database!")
        except Exception as e:
            print(e)


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

