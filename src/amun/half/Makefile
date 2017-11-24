#!/bin/bash

.secondary:

all: gemm_fp16 gemm_normal memcpy_sm50 memcpy_sm35

gemm_fp16: fp16gemm.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11

gemm_normal: gemm.cu
	nvcc $< -arch sm_61 -lcublas -lcurand -o $@ --std=c++11

memcpy_sm50: memcpy.cu
	nvcc $< -arch sm_50 -lcublas -lcurand -o $@ --std=c++11

memcpy_sm35: memcpy.cu
	nvcc $< -arch sm_35 -lcublas -lcurand -o $@ --std=c++11
