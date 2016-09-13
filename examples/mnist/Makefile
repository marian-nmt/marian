.PHONY: all clean download

all: download

download: train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx3-ubyte.gz

%.gz:
	wget http://yann.lecun.com/exdb/mnist/$*.gz -O $@

clean:
	rm -f *.gz
