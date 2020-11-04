#!/bin/sh
echo Building Triton Marian backend ...

docker build -t triton-marian-build .

echo Copying artifacts ...

docker container create --name extract triton-marian-build
docker container cp extract:/opt/tritonserver/marian_backend/build/libtriton_marian.so .
docker container rm -f extract
