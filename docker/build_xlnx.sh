#!/bin/bash

#!/bin/bash

uid=`id -u $USER`
gid=`id -g $USER`

sudo docker build -t xlnx_cred --build-arg user=$USER --build-arg uid=$uid --build-arg gid=$gid --build-arg http_proxy=http://proxy:8080 --build-arg https_proxy=http://proxy:8080 -f Dockerfile.cred . &&
sudo docker build -t $USER/keras --build-arg BASE_IMG=xlnx_cred -f Dockerfile .
