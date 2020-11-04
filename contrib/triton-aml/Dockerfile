# It is recommended to use a machine which supports CUDA to build this image.
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 AS BUILDER
RUN apt-get update --fix-missing
RUN apt-get install -y curl git autoconf automake libtool curl make g++ unzip cmake build-essential cpio
RUN apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

# install zlib
WORKDIR /
RUN git clone --no-checkout https://github.com/madler/zlib
WORKDIR /zlib
RUN git checkout tags/v1.2.10 && \
    ./configure && \
    make install

# protobuf install
WORKDIR /
RUN git clone --no-checkout https://github.com/protocolbuffers/protobuf.git
WORKDIR /protobuf
RUN git checkout tags/v3.8.0 && \
    git submodule update --init --recursive && \
    ./autogen.sh
RUN ./configure --disable-shared --prefix=/usr CFLAGS="-fPIC"  CXXFLAGS="-fPIC" && \
    make && \
    make check  && \
    make install  && \
    ldconfig # refresh shared library cache.

# Intel mkl install
WORKDIR /
RUN curl --tlsv1.2 --output l_mkl_2020.0.166.tgz https://registrationcenter-download.intel.com/akdlm/irc_nas/tec/16318/l_mkl_2020.0.166.tgz
RUN tar zxvf l_mkl_2020.0.166.tgz
WORKDIR /l_mkl_2020.0.166
RUN ./install.sh --silent ./silent.cfg --install_dir /opt/intel/ --accept_eula

# boost install
WORKDIR /
RUN git clone --recursive https://github.com/boostorg/boost --branch boost-1.72.0 /boost
WORKDIR /boost
RUN ./bootstrap.sh
RUN ./b2 install --prefix=/usr --with-system --with-thread --with-date_time --with-regex --with-serialization

# Marian install
WORKDIR /
RUN git clone --no-checkout https://github.com/marian-nmt/marian-dev
WORKDIR marian-dev
RUN git checkout youki/quantize-embedding
RUN git checkout dad48865fd3b7f1d7b891de81040f7651e824510
RUN mkdir src/static
RUN mkdir build
COPY src/cmarian.cpp /marian-dev/src/static
COPY src/logging.cpp /marian-dev/src/common
RUN rm src/CMakeLists.txt
COPY src/CMakeLists.txt /marian-dev/src

WORKDIR /marian-dev/build
RUN cmake .. -DCOMPILE_CPU=on -DCOMPILE_CUDA=on -DUSE_SENTENCEPIECE=on -DUSE_STATIC_LIBS=off -DCOMPILE_SERVER=off -DUSE_FBGEMM=on -DCUDA_cublas_device_LIBRARY=/usr/lib/x86_64-linux-gnu/libcublas.so
RUN make -j $(grep -c ^processor /proc/cpuinfo)

# build cmarian static library
FROM nvcr.io/nvidia/tritonserver:20.09-py3
RUN mkdir -p /marian-dev/build/src/3rd_party/sentencepiece/src
COPY --from=BUILDER /usr/lib/libprotobuf.a /usr/lib
COPY --from=BUILDER /usr/lib/libboost_system.a /usr/lib
COPY --from=BUILDER /marian-dev/build/src/3rd_party/fbgemm/libfbgemm.a /usr/lib
COPY --from=BUILDER /marian-dev/build/src/3rd_party/fbgemm/asmjit/libasmjit.a /usr/lib
COPY --from=BUILDER /marian-dev/build/src/3rd_party/sentencepiece/src/libsentencepiece_train.a /usr/lib
COPY --from=BUILDER /marian-dev/build/src/3rd_party/sentencepiece/src/libsentencepiece.a /usr/lib
COPY --from=BUILDER /marian-dev/build/libmarian.a /usr/lib/libcmarian.a
COPY --from=BUILDER /marian-dev/build/src/libmarian_cuda.a /usr/lib/libcmarian_cuda.a

# build triton custom backend
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            build-essential \
            git \
            libopencv-dev \
            libopencv-core-dev \
            libssl-dev \
            libtool \
            pkg-config \
            rapidjson-dev

# install cmake-3.19.0
RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.0-rc1/cmake-3.19.0-rc1-Linux-x86_64.sh
RUN sh cmake-3.19.0-rc1-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir

ADD marian_backend /opt/tritonserver/marian_backend
WORKDIR /opt/tritonserver/marian_backend
RUN mkdir build
RUN cd build && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install .. && \
    make install
