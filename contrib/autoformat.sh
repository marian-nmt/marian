#!/bin/bash

find ./src -path ./src/3rd_party -path ./src/tests -path ./src/models/experimental -prune -o -iname *.h -o -iname *.cpp -o -iname *.cu | xargs clang-format-3.8 -i
