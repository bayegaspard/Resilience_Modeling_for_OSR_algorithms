# src/nlp/receiver.py

import sys
import json
import numpy as np

from padec import init, generate
initialized = False
init()

def initialize_function():
    global initialized
    if not initialized:
        print("Initializing function in receiver.")
        initialized = True

def receive_messages():
    while True:
        message = sys.stdin.buffer.read()
        if not message:
            break;

        # Decode the JSON message back to a string
        decoded_message = json.loads(message)

        if decoded_message == "__INITIALIZE_FUNCTION__" and initialized == False:
            initialize_function()
        elif decoded_message == "__STOP_RECEIVING__":
            print("Received stop signal. Stopping receiver.")
            break
        elif isinstance(decoded_message, list):
            # Decode the JSON message to a 2D matrix
            matrix = np.array(decoded_message)
            print(matrix.shape)
            fake_header = "5e0e8bffb00b581122d37761080045000034000040004006ef4386580d1f8efb"
            for string in matrix:
                generate(fake_header, string)



if __name__ == '__main__':
    receive_messages()