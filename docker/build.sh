#!/bin/bash

user=root
uid=`id -u root`
gid=100

sudo docker build -t f1_cred --build-arg user=$user --build-arg uid=$uid --build-arg gid=$gid -f Dockerfile.cred . &&

sudo docker build -t $user/keras --build-arg BASE_IMG=f1_cred -f Dockerfile .
