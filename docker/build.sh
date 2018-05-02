user=root
uid=`id -u root`
gid=100

sudo docker build -t root_cred --build-arg user=$user --build-arg uid=$uid --build-arg gid=$gid -f Dockerfile.cred . &&

sudo docker build -t $user/keras --build-arg BASE_IMG=root_cred -f Dockerfile .

