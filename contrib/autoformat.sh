#!/bin/bash -v

if ! [ -x "$( command -v clang-format )" ]
then
    mkdir -p $HOME/.local
    wget -O- http://releases.llvm.org/6.0.0/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz | tar --xz -xf - -C $HOME/.local --strip 1
fi

find ./src \( -path ./src/3rd_party -o -path ./src/tests -o -path ./src/models/experimental \) -prune -o -iname *.h -o -iname *.cpp -o -iname *.cu | xargs clang-format -i
