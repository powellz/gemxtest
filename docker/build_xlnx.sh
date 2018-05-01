#!/bin/bash

uid=`id -u $USER`
gid=`id -g $USER`

sudo docker build -t xlnx_cred --build-arg user=$uid --build-arg uid=$uid --build-arg gid=$gid -f Dockerfile.xlnx_cred . &&
sudo docker build -t $uid/keras --build-arg BASE_IMG=xlnx_cred -f Dockerfile .
