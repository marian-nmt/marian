#!/bin/bash -v

cd data

# get En-De training data for WMT17
wget -nc http://www.statmt.org/europarl/v7/de-en.tgz
wget -nc http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
wget -nc http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz

# extract data
tar -xf de-en.tgz
tar -xf training-parallel-commoncrawl.tgz
tar -xf training-parallel-nc-v12.tgz

# create corpus files
cat europarl-v7.de-en.de commoncrawl.de-en.de training/news-commentary-v12.de-en.de > corpus.de
cat europarl-v7.de-en.en commoncrawl.de-en.en training/news-commentary-v12.de-en.en > corpus.en

# clean
rm -r europarl-* commoncrawl.* training/ *.tgz

cd ..
