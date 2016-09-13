.PHONY: all clean download

all: download

download: train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

%-ubyte: %-ubyte.gz
	gzip -d < $^ > $@

%-ubyte.gz:
	wget http://yann.lecun.com/exdb/mnist/$*.gz -O $@

clean:
	rm -f *.gz
