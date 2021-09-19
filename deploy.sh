#!/bin/sh

set -e

ssh ubuntu@dict.peryaudo.org sudo apt update
ssh ubuntu@dict.peryaudo.org sudo apt install -y python3-pip nginx
ssh ubuntu@dict.peryaudo.org sudo pip3 install gunicorn flask
rsync --info=progress2 --exclude 'images' --exclude '__pycache__' --exclude '*.pdf' -a . ubuntu@dict.peryaudo.org:~/LibreEJ
ssh ubuntu@dict.peryaudo.org sudo cp /home/ubuntu/LibreEJ/libreej-nginx /etc/nginx/sites-available/libreej
ssh ubuntu@dict.peryaudo.org sudo ln -f -s /etc/nginx/sites-available/libreej /etc/nginx/sites-enabled
ssh ubuntu@dict.peryaudo.org sudo cp /home/ubuntu/LibreEJ/libreej.service /etc/systemd/system/
ssh ubuntu@dict.peryaudo.org sudo systemctl daemon-reload
ssh ubuntu@dict.peryaudo.org sudo systemctl restart libreej
ssh ubuntu@dict.peryaudo.org sudo systemctl enable libreej
ssh ubuntu@dict.peryaudo.org sudo systemctl restart nginx
