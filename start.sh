#!/bin/bash
python src/dataLoader/dataLoader.py > log_dataload.txt &
python src/gui/client.py > log_client.txt &
sleep 3
#https://stackoverflow.com/a/38147878
ip=$(grep -oE "http:\/\/127\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+\/" log_client.txt)
# echo $ip
python -m webbrowser "$ip"
# python -c "print('$ip')"