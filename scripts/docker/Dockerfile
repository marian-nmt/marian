FROM ubuntu:16.04
MAINTAINER Tomasz Dwojak <t.dwojak@amu.edu.pl>

RUN apt-get update && apt-get install -y \
	build-essential\
	cmake \
	git \
	libopenblas-base \
	libboost-dev \
	libopenblas-dev \
	python \
	python-dev \
	python-pip \
	wget \
	gfortran \
	g++ \
	unzip \
	zlib1g-dev \
	automake \
	autoconf \
	libtool \
	subversion \
	libboost-all-dev \
	libtcmalloc-minimal4

RUN ln -s /usr/lib/libtcmalloc_minimal.so.4 /usr/lib/libtcmalloc_minimal.so

RUN pip install \
	flask \
	flask-sockets \
	websocket \
	websocket-client

RUN mkdir -p /opt/app
WORKDIR /opt/app
RUN git clone https://github.com/emjotde/amunmt -b cpu

WORKDIR /opt/app/amunmt
RUN mkdir -p release

WORKDIR /opt/app/amunmt/release
RUN cmake .. && make

COPY run.py /opt/app/amunmt/run.py
COPY docker-entrypoint.py /docker-entrypoint.py
RUN chmod +x /docker-entrypoint.py

ENTRYPOINT ["/docker-entrypoint.py"]
CMD ["en-ru"]
