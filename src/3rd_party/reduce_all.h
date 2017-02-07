/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once

#include <device_functions.h>
#include "tensors/tensor.h"

namespace marian {

template <unsigned int blockSize>
__device__ void
reduceBlock(volatile float *sdata, float mySum, const unsigned int tid)
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

template <unsigned int blockSize, bool nIsPow2, class Functor>
__device__ void
reduceBlocks(Functor f, const float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += f(g_idata[i]);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += f(g_idata[i+blockSize]);

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

template <unsigned int blockSize, bool nIsPow2, class Functor>
__global__ void reduceSinglePass(Functor f, const float *g_idata, float *g_odata, unsigned int n)
{

    //
    // PHASE 1: Process all inputs assigned to this block
    //

    reduceBlocks<blockSize, nIsPow2>(f, g_idata, g_odata, n);

    //
    // PHASE 2: Last block finished will process all partial sums
    //

    if (gridDim.x > 1)
    {
        const unsigned int tid = threadIdx.x;
        __shared__ bool amLast;
        extern float __shared__ smem[];

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
            float mySum = 0;

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

template <class Functor>
void ReduceAll(Functor f, Tensor out, Tensor in)
{
    cudaSetDevice(out->getDevice());
    int size = in->shape().elements();
    int threads = std::min(MAX_THREADS, size);
    int blocks  = std::min(MAX_BLOCKS, size / threads  + (size % threads != 0));

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = threads * sizeof(float);

    float* d_idata = in->data();
    float* d_odata = out->data();

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                reduceSinglePass<512, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 256:
                reduceSinglePass<256, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 128:
                reduceSinglePass<128, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 64:
                reduceSinglePass< 64, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 32:
                reduceSinglePass< 32, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 16:
                reduceSinglePass< 16, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  8:
                reduceSinglePass<  8, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  4:
                reduceSinglePass<  4, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  2:
                reduceSinglePass<  2, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  1:
                reduceSinglePass<  1, true><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                reduceSinglePass<512, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 256:
                reduceSinglePass<256, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 128:
                reduceSinglePass<128, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 64:
                reduceSinglePass< 64, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 32:
                reduceSinglePass< 32, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case 16:
                reduceSinglePass< 16, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  8:
                reduceSinglePass<  8, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  4:
                reduceSinglePass<  4, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  2:
                reduceSinglePass<  2, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;

            case  1:
                reduceSinglePass<  1, false><<< dimGrid, dimBlock, smemSize >>>(f, d_idata, d_odata, size);
                break;
        }
    }
}

}
