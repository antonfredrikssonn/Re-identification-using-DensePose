#!/bin/bash
echo "Starting DensePose re-ID docker..."

docker build --tag reid .

docker run --net=host \
    --rm \
    --privileged \
    -u $_UID:$_GID \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -v /etc/sudoers:/etc/sudoers:ro \
    -v /home/bjornel/exjobb/ReID_Using_DensePose:/ReID_Using_DensePose \
    -v $HOME/.gitconfig:$HOME/.gitconfig:ro \
    -v $HOME/.ssh:$HOME/.ssh:ro \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -it \
    reid /ReID_Using_DensePose/scripts/init.sh