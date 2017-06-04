#!/bin/bash

if [ `ls -1 *-ubyte 2>/dev/null | wc -l ` == 4 ]; then
    echo Files exist: `ls -1 *-ubyte`;
    exit;
fi

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gzip -d *-ubyte.gz
