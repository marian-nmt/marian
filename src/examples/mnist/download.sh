#!/bin/bash

if [ `ls -1 *-ubyte 2>/dev/null | wc -l ` == 4 ]; then
    echo Files exist: `ls -1 *-ubyte`;
    exit;
fi

wget https://romang.blob.core.windows.net/mariandev/regression-tests/data/exdb_mnist.tar.gz
tar zxvf exdb_mnist.tar.gz
mv exdb_mnist/*-ubyte .
