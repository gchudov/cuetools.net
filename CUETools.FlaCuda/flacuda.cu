/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#ifndef _FLACUDA_KERNEL_H_
#define _FLACUDA_KERNEL_H_

extern "C" __global__ void cudaComputeAutocor(
    float *output,
    const int *samples,
    const float *window,
    int frameSize,
    int frameOffset,
    int blocks)
{
    __shared__ struct {
	float data[512];
	float matrix[512];
    } shared;
    const int iWin = blockIdx.y >> 2;
    const int iCh = blockIdx.y & 3;
    const int smpBase = iCh * frameOffset;
    const int winBase = iWin * 2 * frameOffset;
    const int pos = blockIdx.x * blocks;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // fetch blockDim.x*blockDim.y samples
    shared.data[tid] = pos + tid < frameSize ? samples[smpBase + pos + tid] * window[winBase + pos + tid] : 0.0f;
    __syncthreads();

    float s = 0.0f;    
    for (int i = 0; i < blocks; i += blockDim.y) 
	s += shared.data[i + threadIdx.y] * shared.data[i + threadIdx.y + threadIdx.x];
    shared.matrix[tid] = s;
    __syncthreads();

    // reduction in shared mem
    for(unsigned int s=blockDim.y/2; s>1; s>>=1)
    {
	if (threadIdx.y < s)
	    shared.matrix[tid] += shared.matrix[tid + s * blockDim.x];
	__syncthreads();
    }

    // return results
    if (threadIdx.y == 0)
	output[(blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x] = shared.matrix[threadIdx.x] + shared.matrix[threadIdx.x + blockDim.x];
}

typedef struct
{
    int residualOrder;
    int shift;
    int coefsOffs;
    int samplesOffs;
} encodeResidualTaskStruct;

extern "C" __global__ void cudaEncodeResidual(
    int*output,
    int*samples,
    int*allcoefs,
    encodeResidualTaskStruct *tasks,
    int frameSize,
    int partSize
    )
{
    __shared__ struct {
	int data[256];
	int residual[256];
	int coefs[32];
	int rice[32];
	encodeResidualTaskStruct task;
    } shared;
    const int tid = threadIdx.x;
    // fetch task data
    if (tid < sizeof(encodeResidualTaskStruct) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    __syncthreads();
    const int pos = blockIdx.x * partSize;
    const int residualOrder = shared.task.residualOrder;
    const int dataLen = min(frameSize - pos, partSize + residualOrder + 1);
    const int residualLen = dataLen - residualOrder - 1;

    // fetch coeffs, inverting their order
    if (tid <= residualOrder) shared.coefs[residualOrder - tid] = allcoefs[shared.task.coefsOffs + tid];
    // fetch samples
    shared.data[tid] = (tid < dataLen ? samples[shared.task.samplesOffs + pos + tid] : 0);
    
    // compute residual
    __syncthreads();
    long sum = 0;
    for (int c = 0; c <= residualOrder; c++)
	sum += __mul24(shared.data[tid + c], shared.coefs[c]);
    int res = shared.data[tid + residualOrder + 1] - (sum >> shared.task.shift);
    shared.residual[tid] = __mul24(tid < residualLen, (2 * res) ^ (res >> 31));
    
    __syncthreads();
    // residual sum: reduction in shared mem
    if (tid < 128) shared.residual[tid] += shared.residual[tid + 128]; __syncthreads();
    if (tid < 64) shared.residual[tid] += shared.residual[tid + 64]; __syncthreads();
    if (tid < 32) shared.residual[tid] += shared.residual[tid + 32]; __syncthreads();
    shared.residual[tid] += shared.residual[tid + 16];
    shared.residual[tid] += shared.residual[tid + 8];
    shared.residual[tid] += shared.residual[tid + 4];
    shared.residual[tid] += shared.residual[tid + 2];
    shared.residual[tid] += shared.residual[tid + 1];
    __syncthreads();

    if (tid < 32)
    {
	// rice parameter search
	shared.rice[tid] = __mul24(tid >= 15, 0x7fffff) + residualLen * (tid + 1) + ((shared.residual[0] - (residualLen >> 1)) >> tid);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 8]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 4]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 2]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 1]);
    }
    if (tid == 0)
	output[blockIdx.x + blockIdx.y * gridDim.x] = shared.rice[0];
}

#if 0
extern "C" __global__ void cudaComputeAutocor3int(const int * samples, const float * window, int* output, int frameSize, int frameOffset, int channels)
{
    extern __shared__ short shared[];
    int *ishared = (int*)shared;
    const int lag = blockIdx.x;
    
    // fetch signal, multiply by window and split high bits/low bits
    const int iWin = blockIdx.y >> 2; // blockIdx.y/channels;
    const int iCh = (blockIdx.y  - iWin * channels);
    const int smpBase = iCh * frameOffset;
    const int winBase = iWin * 2 * frameOffset;

    for(int i = threadIdx.x; i < frameSize; i += blockDim.x)
    {
	float val = samples[smpBase + i] * window[winBase + i];
	int ival = __float2int_rz(fabs(val));
	int sg =  (1 - 2 *signbit(val));
	//int ival = (int) val;
	//int sg =  ival < 0 ? -1 : 1;
	ival = ival < 0 ? -ival : ival;
	shared[i*2] = __mul24(sg, (ival >> 9));
	shared[i*2+1] = __mul24(sg, (ival & ((1 << 9) - 1)));
    }
    __syncthreads();

    // correlation
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    for (int i = threadIdx.x; i < frameSize - lag; i += blockDim.x)
    {
	sum1 += __mul24(shared[2*i], shared[2*(i+lag)]);
	sum2 += __mul24(shared[2*i+1], shared[2*(i+lag)]);
	sum2 += __mul24(shared[2*i], shared[2*(i+lag)+1]);
	sum3 += __mul24(shared[2*i+1], shared[2*(i+lag)+1]);
    }    
    __syncthreads();
    ishared[threadIdx.x] = sum1;
    ishared[threadIdx.x + blockDim.x] = sum2;
    ishared[threadIdx.x + blockDim.x * 2] = sum3;
    __syncthreads();

    // reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (threadIdx.x < s)
        {
            ishared[threadIdx.x] += ishared[threadIdx.x + s];
	    ishared[threadIdx.x + blockDim.x] += ishared[threadIdx.x + s + blockDim.x];
	    ishared[threadIdx.x + blockDim.x * 2] += ishared[threadIdx.x + s + blockDim.x * 2];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
	output[(blockIdx.x + blockIdx.y * gridDim.x) * 3] = ishared[threadIdx.x];
	output[(blockIdx.x + blockIdx.y * gridDim.x) * 3 + 1] = ishared[threadIdx.x + blockDim.x];
	output[(blockIdx.x + blockIdx.y * gridDim.x) * 3 + 2] = ishared[threadIdx.x + blockDim.x * 2];
    } 
}

__device__ float Bartlett(int i, int blockSize)
{
    float n = fminf(i, blockSize - i);
    float k = 2.0f / blockSize * (blockSize / 2.0f - n);
    k = 1.0f - k * k;
    return k*k;
}

extern "C" __global__ void cudaComputeAutocorPart(const int * samples, const float * window, float* output, int frameSize, int frameOffset, int iCh, int iWin)
{
    extern __shared__ float fshared[];
    // fetch signal, multiply by window
    //const int iWin = blockIdx.y;
    //const int iCh = blockIdx.x;
    const int smpBase = iCh * frameOffset;
    const int winBase = iWin * 2 * frameOffset;
    float * matrix = fshared + 129;

    // initialize results matrix
    matrix[threadIdx.x + threadIdx.y * (blockDim.x + 1)] = 0.0f;

    // prefetch blockDim.x + blockDim.y samples
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < blockDim.x + blockDim.y)
    {
	if (blockIdx.x * blockDim.y + tid < frameSize)
	    fshared[tid] = samples[smpBase + blockIdx.x * blockDim.y + tid] * window[winBase + blockIdx.x * blockDim.y + tid];
	else
	    fshared[tid] = 0.0f;
    }

    __syncthreads();

    matrix[threadIdx.x + threadIdx.y * (1 + blockDim.x)] += fshared[threadIdx.y] * fshared[threadIdx.y + threadIdx.x];

    __syncthreads();

    // reduction in shared mem
    for(unsigned int s=blockDim.y/2; s>0; s>>=1)
    {
	if (threadIdx.y < s)
	    matrix[threadIdx.x + threadIdx.y * (1 + blockDim.x)] += matrix[threadIdx.x + (s + threadIdx.y) * (1 + blockDim.x)];
	__syncthreads();
    }

    // return results
    if (threadIdx.y == 0)
	output[blockIdx.x * blockDim.x + threadIdx.x] = matrix[threadIdx.x];
}

extern "C" __global__ void cudaComputeAutocor2(const int * samples, const float * window, float* output, int frameSize, int frameOffset)
{
    extern __shared__ float fshared[];   
    // fetch signal, multiply by window
    const int iWin = blockIdx.y;
    const int iCh = blockIdx.x;
    const int smpBase = iCh * frameOffset;
    const int winBase = iWin * 2 * frameOffset;

    for(int i = threadIdx.x + threadIdx.y * blockDim.x; i < frameSize; i += blockDim.x * blockDim.y)
	fshared[i] = samples[smpBase + i] * window[winBase + i];

    __syncthreads();

    const int lag = threadIdx.y;

    // correlation
    float sum = 0.0f;
    for (int i = threadIdx.x; i < frameSize - lag; i += blockDim.x)
	sum += fshared[i] * fshared[i+lag];
    __syncthreads();

    fshared[threadIdx.x + threadIdx.y * blockDim.x] = sum;

    __syncthreads();

    // reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (threadIdx.x < s)
            fshared[threadIdx.x + threadIdx.y * blockDim.x] += fshared[threadIdx.x + s + threadIdx.y * blockDim.x];
        __syncthreads();
    }

 //   if (threadIdx.x < 32)
 //   {
	//if (blockDim.x >= 64) fshared[threadIdx.x + threadIdx.y * blockDim.x] += fshared[threadIdx.x + 32 + threadIdx.y * blockDim.x];
	//if (blockDim.x >= 32) fshared[threadIdx.x + threadIdx.y * blockDim.x] += fshared[threadIdx.x + 16 + threadIdx.y * blockDim.x];
	//if (blockDim.x >= 16) fshared[threadIdx.x + threadIdx.y * blockDim.x] += fshared[threadIdx.x +  8 + threadIdx.y * blockDim.x];
	//if (blockDim.x >=  8) fshared[threadIdx.x + threadIdx.y * blockDim.x] += fshared[threadIdx.x +  4 + threadIdx.y * blockDim.x];
	//if (blockDim.x >=  4) fshared[threadIdx.x + threadIdx.y * blockDim.x] += fshared[threadIdx.x +  2 + threadIdx.y * blockDim.x];
	//if (blockDim.x >=  2) fshared[threadIdx.x + threadIdx.y * blockDim.x] += fshared[threadIdx.x +  1 + threadIdx.y * blockDim.x];
 //   }

    if (threadIdx.x == 0) {
	output[(blockIdx.x + blockIdx.y * gridDim.x) * blockDim.y + threadIdx.y] 
	= fshared[threadIdx.x + threadIdx.y * blockDim.x];
    } 
}

extern "C" __global__ void cudaComputeAutocorOld(const int * samples, const float * window, float* output, int frameSize, int frameOffset, int channels)
{
    extern __shared__ float fshared[];
    const int lag = blockIdx.x;
    
    // fetch signal, multiply by window
    const int iWin = blockIdx.y >> 2; // blockIdx.y/channels;
    const int iCh = (blockIdx.y  - iWin * channels);
    const int smpBase = iCh * frameOffset;
    const int winBase = iWin * 2 * frameOffset;

    for(int i = threadIdx.x; i < frameSize; i += blockDim.x)
	fshared[i] = samples[smpBase + i] * window[winBase + i];

    __syncthreads();

    // correlation
    float sum = 0.0f;
    for (int i = threadIdx.x; i < frameSize - lag; i += blockDim.x)
	sum += fshared[i] * fshared[i+lag];
    __syncthreads();

    fshared[threadIdx.x] = sum;

    __syncthreads();

    // reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (threadIdx.x < s)
            fshared[threadIdx.x] += fshared[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
	output[blockIdx.x + blockIdx.y * gridDim.x] = fshared[threadIdx.x];
    } 
}
#endif

#endif // _FLACUDA_KERNEL_H_
