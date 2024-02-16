import socket

# Server details
server_ip = '127.0.0.1'
server_port = 8080

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific address and port
server_socket.bind((server_ip, server_port))

# Listen for incoming connections (maximum 1 connection in this example)
server_socket.listen(1)
print(f"Server listening on {server_ip}:{server_port}")

# Accept a connection
client_socket, client_address = server_socket.accept()
print(f"Connection from {client_address}")

# Receive data from the client
received_data = client_socket.recv(1024)

# Print the received data in hexadecimal form
hex_data = received_data.hex()
print(f"Received Hex Data: {hex_data}")

# Close the sockets
client_socket.close()
server_socket.close()
