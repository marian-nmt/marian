#!/bin/bash

if [ ! $# -eq 1 ]; then
  echo "usage: $0 v0.0.1-alpha"
  exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TAG=$1

git tag $TAG
cd $DIR/../build && cmake -DVERSION_UPDATE_FROM_GIT=ON .. && cd $DIR
git add $DIR/../VERSION
git commit -m "Update to version $TAG"
git tag -d $TAG
cd $DIR/../build && cmake -DVERSION_UPDATE_FROM_GIT=OFF .. && cd $DIR
