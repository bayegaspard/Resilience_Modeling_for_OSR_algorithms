#!/bin/bash

# https://unix.stackexchange.com/a/361982
# closeServer() {
#     echo closing $server_id
#     kill $server_id
#     echo closing $client_id
#     kill $client_id
# }
# trap closeServer EXIT

# https://superuser.com/a/562804
trap "kill -- -$$" EXIT


python src/dataLoader/dataLoader.py > log_dataload.txt &
# server_id=($!)
python src/gui/client.py > log_client.txt &
# client_id=($!)
sleep 6
#https://stackoverflow.com/a/38147878
ip=$(grep -oE "http:\/\/127\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+\/" log_client.txt)
# echo $ip
python -m webbrowser "$ip"
# python -c "print('$ip')"

sleep 99999

