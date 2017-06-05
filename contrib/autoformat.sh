#!/bin/bash

find ./src -path ./src/3rd_party -prune -o -iname *.h -o -iname *.cpp | xargs clang-format-3.8 -i
