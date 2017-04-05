#!/bin/bash

.secondary:

all: gemm_fp16 gemm_normal

gemm_fp16: fp16gemm.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11

gemm_normal: gemm.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11
