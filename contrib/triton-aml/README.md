Triton-AML
======

*Triton-AML* is a Triton custom backend running with Marian in the AzureML Inference Environment, it's one of the implementation of [Triton Backend Shared Library](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend.html#backend-shared-library).

This backend is compiled with the static library of Marian on a specific version.

Layout:

- marian_backend: Triton Marian backend source code
- src: Changed code and CMakeLists.txt of Marian
- Dockerfile: Used for compiling the backend with the static library of Marian
- build.sh: A simple shell script to run the Dockerfile to get the generated libtriton_marian.so

## Usage

Run `./build.sh` to get the Triton Marian backend shared library.

For all the users, you can put the libtriton_marian.so into the following places:

- <model_repository>/<model_name>/<version_directory>/libtriton_marian.so
- <model_repository>/<model_name>/libtriton_marian.so

For the AzureML Inference team members, you can put it into the following place of *aml-triton* base image:

- <backend_directory>/marian/libtriton_marian.so

Where <backend_directory> is by default /opt/tritonserver/backends.

## Make changes

If you want to compile with another version of Marian, you need to replace `RUN git checkout youki/quantize-embedding` in the Dockerfile, then copy the new CMakeLists.txt replace the old one, add src/cmarian.cpp into CMakeLists.txt and make some changes to make sure it will build a static library of Marian.

## Limitation

For now, it's only used for *nlxseq2seq* model, some hard code is in the `ModelState::SetMarianConfigPath` function, some changes must be done if you want to run other models with Marian.
