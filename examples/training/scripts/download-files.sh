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

if [ ! -f data/corpus.bt.ro-en.en.gz ];
then
    wget http://data.statmt.org/rsennrich/wmt16_backtranslations/ro-en/corpus.bt.ro-en.en.gz -O data/corpus.bt.ro-en.en.gz
    wget http://data.statmt.org/rsennrich/wmt16_backtranslations/ro-en/corpus.bt.ro-en.ro.gz -O data/corpus.bt.ro-en.ro.gz
fi

cd data/
tar -xf ro-en.tgz
unzip SETIMES2.ro-en.txt.zip
gzip -d corpus.bt.ro-en.en.gz corpus.bt.ro-en.ro.gz

cat europarl-v7.ro-en.en SETIMES2.en-ro.en corpus.bt.ro-en.en > corpus.en
cat europarl-v7.ro-en.ro SETIMES2.en-ro.ro corpus.bt.ro-en.ro > corpus.ro

cd ..
