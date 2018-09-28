#!/bin/bash -x
find ./src -type f -not -path "./src/3rd_party/*" -exec sed -i 's///' {} +
