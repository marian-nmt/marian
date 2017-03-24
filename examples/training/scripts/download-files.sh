#!/bin/bash -v

# get En-Ro training data for WMT16

if [ ! -f data/ro-en.tgz ];
then
  wget http://www.statmt.org/europarl/v7/ro-en.tgz -O data/ro-en.tgz
fi

if [ ! -f data/SETIMES2.ro-en.txt.zip ];
then
  wget http://opus.lingfil.uu.se/download.php?f=SETIMES2/en-ro.txt.zip -O data/SETIMES2.ro-en.txt.zip
fi

cd data/
tar -xf ro-en.tgz
unzip SETIMES2.ro-en.txt.zip

cat europarl-v7.ro-en.en SETIMES2.en-ro.en > corpus.en
cat europarl-v7.ro-en.ro SETIMES2.en-ro.ro > corpus.ro

cd ..
