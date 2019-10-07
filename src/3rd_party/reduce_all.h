// This software contains source code provided by NVIDIA Corporation.

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

/*
MJD: Relevant text from the NVIDIA EULA:

2.1 Sample Source Code Modification, Ownership and Distribution

Subject to the terms of the SLA and this Supplement, NVIDIA hereby grants you a non-
exclusive, non-transferable license, without the right to sublicense, during the applicable
license term unless earlier terminated pursuant to the SLA, to have Authorized Users
modify and create derivative works of CUDA Licensed Software that constitutes sample
source code, when provided to you by NVIDIA in source code form. You hold all rights,
title and interest in and to your modifications and derivative works of the sample source
code software that you create as permitted hereunder (collective, Derivatives”), subject
to NVIDIA’s underlying Intellectual Property Rights in and to the CUDA Licensed
Software; provided, however that you grant NVIDIA and its Affiliates an irrevocable,
perpetual, nonexclusive, worldwide, royalty-free paid-up license to make, have made,
use, have used, reproduce, license, distribute, sublicense, transfer and otherwise
commercialize Derivatives including (without limitation) with the CUDA Licensed
Software or other NVIDIA products, technologies or materials. You may distribute the
CUDA Supplement to Software License Agreement End User License Agreements (EULA) 
DR-06739-001_v01_v9.0 | 14 sample source code as delivered by NVIDIA and/or your Derivatives, 
provided that all NVIDIA copyright notices and trademarks are maintained and used properly 
and the sample source code includes the following notice: “This software contains source code
provided by NVIDIA Corporation.”
*/

#pragma once

#include "tensors/tensor.h"

#include <cuda_runtime.h>

namespace marian {

template <unsigned int blockSize, typename AccType>
__device__ void
reduceBlock(volatile AccType *sdata, AccType mySum, const unsigned int tid)
{
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        if (blockSize >=  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 32];
        }

        if (blockSize >=  32)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 16];
        }

        if (blockSize >=  16)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  8];
        }

        if (blockSize >=   8)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  4];
        }

        if (blockSize >=   4)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  2];
        }

        if (blockSize >=   2)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  1];
        }
    }
}

template <unsigned int blockSize, bool nIsPow2, typename T, typename AccType, class Functor>
__device__ void
reduceBlocks(Functor f, T *g_idata, AccType *g_odata, unsigned int n)
{
    extern __shared__ AccType sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    AccType mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += f((AccType)g_idata[i]);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += f((AccType)g_idata[i+blockSize]);

        i += gridSize;
    }

    // do reduction in shared mem
    reduceBlock<blockSize>(sdata, mySum, tid);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Global variable used by reduceSinglePass to count how many blocks have finished
__device__ unsigned int retirementCount = 0;

cudaError_t setRetirementCount(int retCnt)
{
    return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}

// This reduction kernel reduces an arbitrary size array in a single kernel invocation
// It does so by keeping track of how many blocks have finished.  After each thread
// block completes the reduction of its own block of data, it "takes a ticket" by
// atomically incrementing a global counter.  If the ticket value is equal to the number
// of thread blocks, then the block holding the ticket knows that it is the last block
// to finish.  This last block is responsible for summing the results of all the other
// blocks.
//
// In order for this to work, we must be sure that before a block takes a ticket, all
// of its memory transactions have completed.  This is what __threadfence() does -- it
// blocks until the results of all outstanding memory transactions within the
// calling thread are visible to all other threads.
//
// For more details on the reduction algorithm (notably the multi-pass approach), see
// the "reduction" sample in the CUDA SDK.

template <unsigned int blockSize, bool nIsPow2, typename T, typename AccType, class Functor>
__global__ void reduceSinglePass(Functor f, T *g_idata, AccType *g_odata, unsigned int n)
{

    //
    // PHASE 1: Process all inputs assigned to this block
    //

    reduceBlocks<blockSize, nIsPow2, T, AccType>(f, g_idata, g_odata, n);

    //
    // PHASE 2: Last block finished will process all partial sums
    //

    if (gridDim.x > 1)
    {
        const unsigned int tid = threadIdx.x;
        __shared__ bool amLast;
        extern AccType __shared__ smem[];

        // wait until all outstanding memory instructions in this thread are finished
        __threadfence();

        // Thread 0 takes a ticket
        if (tid==0)
        {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            // If the ticket ID is equal to the number of blocks, we are the last block!
            amLast = (ticket == gridDim.x-1);
        }

        __syncthreads();

        // The last block sums the results of all other blocks
        if (amLast)
        {
            int i = tid;
            AccType mySum = 0;

            while (i < gridDim.x)
            {
                mySum += g_odata[i];
                i += blockSize;
            }

            reduceBlock<blockSize>(smem, mySum, tid);

            if (tid==0)
            {
                g_odata[0] = smem[0];

                // reset retirement count so that next run succeeds
                retirementCount = 0;
            }
        }
    }
}

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

template <typename T, typename AccType, class Functor>
void ReduceAll(Functor f, Tensor blockMem, Tensor in)
{
    cudaSetDevice(in->getDeviceId().no);
    int size = in->shape().elements();
    int threads = std::min(MAX_THREADS, size);
    int blocks  = std::min(MAX_BLOCKS, size / threads  + (size % threads != 0));

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = threads * sizeof(AccType);

    T* d_idata = in->data<T>();
    AccType* d_odata = blockMem->data<AccType>();

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                reduceSinglePass<512, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 256:
                reduceSinglePass<256, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 128:
                reduceSinglePass<128, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 64:
                reduceSinglePass< 64, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 32:
                reduceSinglePass< 32, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 16:
                reduceSinglePass< 16, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  8:
                reduceSinglePass<  8, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  4:
                reduceSinglePass<  4, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  2:
                reduceSinglePass<  2, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  1:
                reduceSinglePass<  1, true, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduceSinglePass<512, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 256:
                reduceSinglePass<256, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 128:
                reduceSinglePass<128, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 64:
                reduceSinglePass< 64, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 32:
                reduceSinglePass< 32, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 16:
                reduceSinglePass< 16, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  8:
                reduceSinglePass<  8, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  4:
                reduceSinglePass<  4, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  2:
                reduceSinglePass<  2, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  1:
                reduceSinglePass<  1, false, T, AccType><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;
        }
    }
}

}
