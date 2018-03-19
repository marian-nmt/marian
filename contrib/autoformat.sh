#!/bin/bash

find ./src \( -path ./src/3rd_party -o -path ./src/tests -o -path ./src/models/experimental \) -prune -o -iname *.h -o -iname *.cpp -o -iname *.cu | xargs clang-format -i
