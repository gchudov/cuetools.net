/**
 * CUETools.FlaCuda: FLAC audio encoder using CUDA
 * Copyright (c) 2009 Gregory S. Chudov
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef _FLACUDA_KERNEL_H_
#define _FLACUDA_KERNEL_H_

typedef enum
{
    Constant = 0,
    Verbatim = 1,
    Fixed = 8,
    LPC = 32
} SubframeType;

typedef struct
{
    int residualOrder; // <= 32
    int samplesOffs;
    int shift;
    int cbits;
    int size;
    int type;
    int obits;
    int blocksize;
    int best_index;
    int channel;
    int residualOffs;
    int wbits;
    int abits;
    int porder;
    int reserved[2];
} FlaCudaSubframeData;

typedef struct
{
    FlaCudaSubframeData data;
    int coefs[32]; // fixme: should be short?
} FlaCudaSubframeTask;

#define SUM16(buf,tid,op)   buf[tid] op buf[tid + 8]; buf[tid] op buf[tid + 4]; buf[tid] op buf[tid + 2]; buf[tid] op buf[tid + 1];
#define SUM32(buf,tid,op)   buf[tid] op buf[tid + 16]; SUM16(buf,tid,op)
#define SUM64(buf,tid,op)   if (tid < 32) buf[tid] op buf[tid + 32]; __syncthreads(); if (tid < 32) { SUM32(buf,tid,op) }
#define SUM128(buf,tid,op)  if (tid < 64) buf[tid] op buf[tid + 64]; __syncthreads(); SUM64(buf,tid,op)
#define SUM256(buf,tid,op)  if (tid < 128) buf[tid] op buf[tid + 128]; __syncthreads(); SUM128(buf,tid,op)
#define SUM512(buf,tid,op)  if (tid < 256) buf[tid] op buf[tid + 256]; __syncthreads(); SUM256(buf,tid,op)

#define FSQR(s) ((s)*(s))
#define FASTMUL(a,b) __mul24(a,b)

extern "C" __global__ void cudaStereoDecorr(
    int *samples,
    short2 *src,
    int offset
)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < offset)
    {
	short2 s = src[pos];
	samples[pos] = s.x;
	samples[1 * offset + pos] = s.y;
	samples[2 * offset + pos] = (s.x + s.y) >> 1;
	samples[3 * offset + pos] = s.x - s.y;
    }
}

extern "C" __global__ void cudaChannelDecorr2(
    int *samples,
    short2 *src,
    int offset
)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < offset)
    {
	short2 s = src[pos];
	samples[pos] = s.x;
	samples[1 * offset + pos] = s.y;
    }
}

extern "C" __global__ void cudaChannelDecorr(
    int *samples,
    short *src,
    int offset
)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < offset)
	samples[blockIdx.y * offset + pos] = src[pos * gridDim.y + blockIdx.y];
}

extern "C" __global__ void cudaFindWastedBits(
    FlaCudaSubframeTask *tasks,
    int *samples,
    int tasksPerChannel
)
{
    __shared__ struct {
	volatile int wbits[256];
	volatile int abits[256];
	FlaCudaSubframeData task;
    } shared;

    if (threadIdx.x < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[threadIdx.x] = ((int*)(&tasks[blockIdx.x * tasksPerChannel].data))[threadIdx.x];
    shared.wbits[threadIdx.x] = 0;
    shared.abits[threadIdx.x] = 0;
    __syncthreads();

    for (int pos = 0; pos < shared.task.blocksize; pos += blockDim.x)
    {
	int smp = pos + threadIdx.x < shared.task.blocksize ? samples[shared.task.samplesOffs + pos + threadIdx.x] : 0;
	shared.wbits[threadIdx.x] |= smp;
	shared.abits[threadIdx.x] |= smp ^ (smp >> 31);
    }
    __syncthreads();
    SUM256(shared.wbits, threadIdx.x, |=);
    SUM256(shared.abits, threadIdx.x, |=);
    if (threadIdx.x == 0)
	shared.task.wbits = max(0,__ffs(shared.wbits[0]) - 1);
    if (threadIdx.x == 0)
	shared.task.abits = 32 - __clz(shared.abits[0]) - shared.task.wbits;
    __syncthreads();

    if (threadIdx.x < tasksPerChannel)
	tasks[blockIdx.x * tasksPerChannel + threadIdx.x].data.wbits = shared.task.wbits;
    if (threadIdx.x < tasksPerChannel)
	tasks[blockIdx.x * tasksPerChannel + threadIdx.x].data.abits = shared.task.abits;
}

extern "C" __global__ void cudaComputeAutocor(
    float *output,
    const int *samples,
    const float *window,
    FlaCudaSubframeTask *tasks,
    const int max_order, // should be <= 32
    const int windowCount, // windows (log2: 0,1)
    const int taskCount // tasks per block
)
{
    __shared__ struct {
	float data[512];
	volatile float product[256];
	FlaCudaSubframeData task;
	volatile float result[33];
	volatile int dataPos;
	volatile int dataLen;
	volatile int windowOffs;
	volatile int samplesOffs;
	//volatile int resultOffs;
    } shared;
    const int tid = threadIdx.x + (threadIdx.y * 32);
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + __mul24(taskCount, blockIdx.y >> windowCount)))[tid];
    if (tid == 0) 
    {
	shared.dataPos = __mul24(blockIdx.x, 15 * 32);
	shared.windowOffs = __mul24(blockIdx.y & ((1 << windowCount)-1), shared.task.blocksize) + shared.dataPos;
	shared.samplesOffs = shared.task.samplesOffs + shared.dataPos;
	shared.dataLen = min(shared.task.blocksize - shared.dataPos, 15 * 32 + max_order);
    }
    //if (tid == 32)
	//shared.resultOffs = __mul24(blockIdx.x + __mul24(blockIdx.y, gridDim.x), max_order + 1);
    __syncthreads();

    // fetch samples
    shared.data[tid] = tid < shared.dataLen ? samples[shared.samplesOffs + tid] * window[shared.windowOffs + tid]: 0.0f;
    int tid2 = tid + 256;
    shared.data[tid2] = tid2 < shared.dataLen ? samples[shared.samplesOffs + tid2] * window[shared.windowOffs + tid2]: 0.0f;
    __syncthreads();

    const int ptr = __mul24(threadIdx.x, 15);
    for (int lag = threadIdx.y; lag <= max_order; lag += 8)
    {
        //const int productLen = min(shared.task.blocksize - blockIdx.x * partSize - lag, partSize);
	const int ptr2 = ptr + lag;
	shared.product[tid] =
	    shared.data[ptr + 0] * shared.data[ptr2 + 0] +
	    shared.data[ptr + 1] * shared.data[ptr2 + 1] +
	    shared.data[ptr + 2] * shared.data[ptr2 + 2] +
	    shared.data[ptr + 3] * shared.data[ptr2 + 3] +
	    shared.data[ptr + 4] * shared.data[ptr2 + 4] +
	    shared.data[ptr + 5] * shared.data[ptr2 + 5] +
	    shared.data[ptr + 6] * shared.data[ptr2 + 6] +
	    shared.data[ptr + 7] * shared.data[ptr2 + 7] +
	    shared.data[ptr + 8] * shared.data[ptr2 + 8] +
	    shared.data[ptr + 9] * shared.data[ptr2 + 9] +
	    shared.data[ptr + 10] * shared.data[ptr2 + 10] +
	    shared.data[ptr + 11] * shared.data[ptr2 + 11] +
	    shared.data[ptr + 12] * shared.data[ptr2 + 12] +
	    shared.data[ptr + 13] * shared.data[ptr2 + 13] +
	    shared.data[ptr + 14] * shared.data[ptr2 + 14];
	shared.product[tid] = shared.product[tid] + shared.product[tid + 8] + shared.product[tid + 16] + shared.product[tid + 24];
	shared.product[tid] = shared.product[tid] + shared.product[tid + 2] + shared.product[tid + 4] + shared.product[tid + 6];
	// return results
	if (threadIdx.x == 0)
	    shared.result[lag] = shared.product[tid] + shared.product[tid + 1];
    }
    __syncthreads();
    if (tid <= max_order)
	output[__mul24(blockIdx.x + __mul24(blockIdx.y, gridDim.x), max_order + 1) + tid] = shared.result[tid];
}

extern "C" __global__ void cudaComputeLPC(
    FlaCudaSubframeTask *tasks,
    int taskCount, // tasks per block
    float*autoc,
    int max_order, // should be <= 32
    float *lpcs,
    int windowCount,
    int partCount
)
{
    __shared__ struct {
	FlaCudaSubframeData task;
	volatile float parts[32];
	volatile float ldr[32];
	volatile float gen1[32];
	volatile float error[32];
	volatile float autoc[33];
	volatile int lpcOffs;
	volatile int autocOffs;
    } shared;
    const int tid = threadIdx.x;// + threadIdx.y * 32;
    
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y * taskCount))[tid];
    if (tid == 0)
    {
	shared.lpcOffs = __mul24(blockIdx.x + __mul24(blockIdx.y, windowCount), max_order + 1) * 32;
	shared.autocOffs = __mul24(__mul24(blockIdx.x + __mul24(blockIdx.y, gridDim.x), max_order + 1), partCount);
    }
    //__syncthreads();
    
    // add up autocorrelation parts

 //   for (int order = threadIdx.x; order <= max_order; order += 32)
 //   {
	//float sum = 0.0f;
	//for (int pos = 0; pos < partCount; pos++)
	//    sum += autoc[shared.autocOffs + pos * (max_order + 1) + order];
	//shared.autoc[order] = sum;
 //   }

    for (int order = 0; order <= max_order; order ++)
    {
	shared.parts[tid] = 0.0f;
	for (int pos = threadIdx.x; pos < partCount; pos += 32)
	    shared.parts[tid] += autoc[shared.autocOffs + pos * (max_order + 1) + order];
	shared.parts[tid] = shared.parts[tid] + shared.parts[tid + 8] + shared.parts[tid + 16] + shared.parts[tid + 24];
	shared.parts[tid] = shared.parts[tid] + shared.parts[tid + 2] + shared.parts[tid + 4] + shared.parts[tid + 6];
	if (threadIdx.x == 0)
	    shared.autoc[order] = shared.parts[tid] + shared.parts[tid + 1];
    }
    //__syncthreads();

    // Compute LPC using Schur and Levinson-Durbin recursion
    if (threadIdx.y == 0)
    {
	float gen0 = shared.gen1[threadIdx.x] = shared.autoc[threadIdx.x+1];
	shared.ldr[threadIdx.x] = 0.0f;
	float error = shared.autoc[0];
	for (int order = 0; order < max_order; order++)
	{
	    // Schur recursion
	    float reff = -shared.gen1[0] / error;
	    error += shared.gen1[0] * reff; // Equivalent to error *= (1 - reff * reff);
    
	    if (threadIdx.x < max_order - 1 - order)
	    {
		float gen1 = shared.gen1[threadIdx.x + 1] + reff * gen0;
		gen0 += shared.gen1[threadIdx.x + 1] * reff;
		shared.gen1[threadIdx.x] = gen1;
	    }

	    // Store prediction error
	    if (threadIdx.x == 0)
		shared.error[order] = error;

	    // Levinson-Durbin recursion
	    shared.ldr[threadIdx.x] += (threadIdx.x < order) * reff * shared.ldr[order - 1 - threadIdx.x] + (threadIdx.x  == order) * reff;

	    // Output coeffs
	    if (threadIdx.x <= order)
		lpcs[shared.lpcOffs + order * 32 + threadIdx.x] = -shared.ldr[order - threadIdx.x];
	}
	// Output prediction error estimates
	if (threadIdx.x < max_order)
	    lpcs[shared.lpcOffs + max_order * 32 + threadIdx.x] = shared.error[threadIdx.x];
    }
}

extern "C" __global__ void cudaComputeLPCLattice(
    FlaCudaSubframeTask *tasks,
    const int taskCount, // tasks per block
    const int *samples,
    const int windowCount,
    const int max_order, // should be <= 12
    float*lpcs
)
{
    __shared__ struct {
	volatile FlaCudaSubframeData task;
	volatile float F[512];
	volatile float arp[32];
	volatile float tmp[256];
	volatile float error[32];
	volatile int lpcOffs;
    } shared;

    // fetch task data
    if (threadIdx.x < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[threadIdx.x] = ((int*)(tasks + taskCount * blockIdx.y))[threadIdx.x];
    if (threadIdx.x == 0)
	shared.lpcOffs = __mul24(__mul24(blockIdx.y + 1, windowCount) - 1, max_order + 1) * 32;
    __syncthreads();

    // F = samples; B = samples
    float s1 = threadIdx.x < shared.task.blocksize ? (samples[shared.task.samplesOffs + threadIdx.x]) / 32768.0f : 0.0f;
    float s2 = threadIdx.x + 256 < shared.task.blocksize ? (samples[shared.task.samplesOffs + threadIdx.x + 256]) / 32768.0f : 0.0f;
    shared.F[threadIdx.x] = s1;
    shared.F[threadIdx.x + 256] = s2;
    __syncthreads();

    shared.tmp[threadIdx.x] = FSQR(s1) + FSQR(s2);
    __syncthreads();
    SUM256(shared.tmp, threadIdx.x, +=);
    __syncthreads();
    float DEN = shared.tmp[0];
    __syncthreads();

    for (int order = 0; order < max_order; order++)
    {
	// reff = F(order+1:frameSize) * B(1:frameSize-order)' / DEN
	int idxF = threadIdx.x + order + 1;
	int idxF2 = idxF + 256;

	shared.tmp[threadIdx.x] = idxF < shared.task.blocksize ? shared.F[idxF] * s1 : 0.0f;
	shared.tmp[threadIdx.x] += idxF2 < shared.task.blocksize ? shared.F[idxF2] * s2 : 0.0f;
	__syncthreads(); 
	SUM256(shared.tmp, threadIdx.x, +=);
	__syncthreads();
	float reff = shared.tmp[0] / DEN;
	__syncthreads();

	// arp(order) = rc(order) = reff
	if (threadIdx.x == 0)
	    shared.arp[order] = reff;
	    //shared.rc[order - 1] = shared.lpc[order - 1][order - 1] = reff;

	// Levinson-Durbin recursion
	// arp(1:order-1) = arp(1:order-1) - reff * arp(order-1:-1:1)
	if (threadIdx.x < order)
	    shared.arp[threadIdx.x] = shared.arp[threadIdx.x] - reff * shared.arp[order - 1 - threadIdx.x];
	
	// Output coeffs
	if (threadIdx.x <= order)
	    lpcs[shared.lpcOffs + order * 32 + threadIdx.x] = shared.arp[order - threadIdx.x];

	// F1 = F(order+1:frameSize) - reff * B(1:frameSize-order)
	// B(1:frameSize-order) = B(1:frameSize-order) - reff * F(order+1:frameSize)
	// F(order+1:frameSize) = F1
	if (idxF < shared.task.blocksize)
	{
	    float f1 = shared.F[idxF];
	    shared.F[idxF] -= reff * s1;
	    s1 -= reff * f1;
	}
	if (idxF2 < shared.task.blocksize)
	{
	    float f2 = shared.F[idxF2];
	    shared.F[idxF2] -= reff * s2;
	    s2 -= reff * f2;
	}

	// DEN = F(order+1:frameSize) * F(order+1:frameSize)' + B(1:frameSize-order) * B(1:frameSize-order)' (BURG)
	shared.tmp[threadIdx.x] = (idxF + 1 < shared.task.blocksize ? FSQR(shared.F[idxF]) + FSQR(s1) : 0);
	shared.tmp[threadIdx.x] += (idxF2 + 1 < shared.task.blocksize ? FSQR(shared.F[idxF2]) + FSQR(s2) : 0);
	__syncthreads();
	SUM256(shared.tmp, threadIdx.x, +=);
	__syncthreads();
	DEN = shared.tmp[0] / 2;
	// shared.PE[order-1] = shared.tmp[0] / 2 / (frameSize - order + 1);
	if (threadIdx.x == 0)
	    shared.error[order] = DEN / (shared.task.blocksize - order);
	__syncthreads();
    }

    // Output prediction error estimates
    if (threadIdx.x < max_order)
	lpcs[shared.lpcOffs + max_order * 32 + threadIdx.x] = shared.error[threadIdx.x];
}

extern "C" __global__ void cudaQuantizeLPC(
    FlaCudaSubframeTask *tasks,
    int taskCount, // tasks per block
    int taskCountLPC, // tasks per set of coeffs (<= 32)
    float*lpcs,
    int max_order, // should be <= 32
    int minprecision,
    int precisions
)
{
    __shared__ struct {
	FlaCudaSubframeData task;
	volatile int tmpi[128];
	volatile int index[64];
	volatile float error[64];
	volatile int lpcOffs;
    } shared;

    if (threadIdx.y == 0)
    {
	// fetch task data
	if (threadIdx.x < sizeof(shared.task) / sizeof(int))
	    ((int*)&shared.task)[threadIdx.x] = ((int*)(tasks + blockIdx.y * taskCount))[threadIdx.x];
	if (threadIdx.x == 0)
	    shared.lpcOffs = (blockIdx.x + blockIdx.y * gridDim.x) * (max_order + 1) * 32;

	shared.index[threadIdx.x] = min(max_order - 1, threadIdx.x);
	shared.error[threadIdx.x] = shared.task.blocksize * 64 + threadIdx.x;
	shared.index[32 + threadIdx.x] = min(max_order - 1, threadIdx.x);
	shared.error[32 + threadIdx.x] = shared.task.blocksize * 64 + threadIdx.x;

        // Select best orders based on Akaike's Criteria

	// Load prediction error estimates
	if (threadIdx.x < max_order)
	    shared.error[threadIdx.x] = shared.task.blocksize * __logf(lpcs[shared.lpcOffs + max_order * 32 + threadIdx.x]) + threadIdx.x * 5.12f * __logf(shared.task.blocksize);
	    //shared.error[threadIdx.x] = shared.task.blocksize * __logf(lpcs[shared.lpcOffs + max_order * 32 + threadIdx.x]) + threadIdx.x * 0.30f * (shared.task.abits + 1) * __logf(shared.task.blocksize);

	// Sort using bitonic sort
	for(int size = 2; size < 64; size <<= 1){
	    //Bitonic merge
	    int ddd = (threadIdx.x & (size / 2)) == 0;
	    for(int stride = size / 2; stride > 0; stride >>= 1){
		//__syncthreads();
		int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
		if ((shared.error[pos] >= shared.error[pos + stride]) == ddd)
		{
		    float t = shared.error[pos];
		    shared.error[pos] = shared.error[pos + stride];
		    shared.error[pos + stride] = t;
		    int t1 = shared.index[pos];
		    shared.index[pos] = shared.index[pos + stride];
		    shared.index[pos + stride] = t1;
		}
	    }
	}

	//ddd == dir for the last bitonic merge step
	{
	    for(int stride = 32; stride > 0; stride >>= 1){
		//__syncthreads();
		int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
		if (shared.error[pos] >= shared.error[pos + stride])
		{
		    float t = shared.error[pos];
		    shared.error[pos] = shared.error[pos + stride];
		    shared.error[pos + stride] = t;
		    int t1 = shared.index[pos];
		    shared.index[pos] = shared.index[pos + stride];
		    shared.index[pos + stride] = t1;
		}
	    }
	}
    }
    __syncthreads();

    const int tid = threadIdx.x + threadIdx.y * 32;

    // Quantization
    for (int i = threadIdx.y; i < taskCountLPC; i += 4)
    {
	int order = shared.index[i >> precisions];
	float lpc = threadIdx.x <= order ? lpcs[shared.lpcOffs + order * 32 + threadIdx.x] : 0.0f;
	// get 15 bits of each coeff
	int coef = __float2int_rn(lpc * (1 << 15));
	// remove sign bits
	shared.tmpi[tid] = coef ^ (coef >> 31);
	// OR reduction
	shared.tmpi[tid] = shared.tmpi[tid] | shared.tmpi[tid + 8] | shared.tmpi[tid + 16] | shared.tmpi[tid + 24];
	shared.tmpi[tid] = shared.tmpi[tid] | shared.tmpi[tid + 2] | shared.tmpi[tid + 4] | shared.tmpi[tid + 6];
	//SUM32(shared.tmpi,tid,|=);
	// choose precision	
	//int cbits = max(3, min(10, 5 + (shared.task.abits >> 1))); //  - __float2int_rn(shared.PE[order - 1])
	int cbits = max(3, min(min(13 - minprecision + (i - ((i >> precisions) << precisions)) - (shared.task.blocksize <= 2304) - (shared.task.blocksize <= 1152) - (shared.task.blocksize <= 576), shared.task.abits), __clz(order) + 1 - shared.task.abits));
	// calculate shift based on precision and number of leading zeroes in coeffs
	int shift = max(0,min(15, __clz(shared.tmpi[threadIdx.y * 32] | shared.tmpi[threadIdx.y * 32 + 1]) - 18 + cbits));
	//if (shared.task.abits + 32 - __clz(order) < shift
	//int shift = max(0,min(15, (shared.task.abits >> 2) - 14 + __clz(shared.tmpi[threadIdx.x & ~31]) + ((32 - __clz(order))>>1)));
	// quantize coeffs with given shift
	coef = max(-(1 << (cbits - 1)), min((1 << (cbits - 1)) -1, __float2int_rn(lpc * (1 << shift))));
	// error correction
	//shared.tmp[threadIdx.x] = (threadIdx.x != 0) * (shared.arp[threadIdx.x - 1]*(1 << shared.task.shift) - shared.task.coefs[threadIdx.x - 1]);
	//shared.task.coefs[threadIdx.x] = max(-(1 << (shared.task.cbits - 1)), min((1 << (shared.task.cbits - 1))-1, __float2int_rn((shared.arp[threadIdx.x]) * (1 << shared.task.shift) + shared.tmp[threadIdx.x])));
	// remove sign bits
	shared.tmpi[tid] = coef ^ (coef >> 31);
	// OR reduction
	shared.tmpi[tid] = shared.tmpi[tid] | shared.tmpi[tid + 8] | shared.tmpi[tid + 16] | shared.tmpi[tid + 24];
	shared.tmpi[tid] = shared.tmpi[tid] | shared.tmpi[tid + 2] | shared.tmpi[tid + 4] | shared.tmpi[tid + 6];
	//SUM32(shared.tmpi,tid,|=);
	// calculate actual number of bits (+1 for sign)
	cbits = 1 + 32 - __clz(shared.tmpi[threadIdx.y * 32] | shared.tmpi[threadIdx.y * 32 + 1]);

	// output shift, cbits and output coeffs
	int taskNo = blockIdx.y * taskCount + blockIdx.x * taskCountLPC + i;
	if (threadIdx.x == 0)
	    tasks[taskNo].data.shift = shift;
	if (threadIdx.x == 0)
	    tasks[taskNo].data.cbits = cbits;
	if (threadIdx.x == 0)
	    tasks[taskNo].data.residualOrder = order + 1;
	if (threadIdx.x <= order)
	    tasks[taskNo].coefs[threadIdx.x] = coef;
    }
}

// blockDim.x == 32
// blockDim.y == 8
extern "C" __global__ void cudaEstimateResidual(
    int*output,
    int*samples,
    FlaCudaSubframeTask *tasks,
    int max_order,
    int partSize // should be blockDim.x * blockDim.y == 256
    )
{
    __shared__ struct {
	int data[32*9];
	volatile int residual[32*8];
	FlaCudaSubframeData task[8];
	int coefs[32*8];
    } shared;
    const int tid = threadIdx.x + threadIdx.y * 32;
    if (threadIdx.x < sizeof(FlaCudaSubframeData)/sizeof(int))
	((int*)&shared.task[threadIdx.y])[threadIdx.x] = ((int*)(&tasks[blockIdx.y * blockDim.y + threadIdx.y]))[threadIdx.x];
    __syncthreads();
    const int pos = blockIdx.x * partSize;
    const int dataLen = min(shared.task[0].blocksize - pos, partSize + max_order);

    // fetch samples
    shared.data[tid] = tid < dataLen ? samples[shared.task[0].samplesOffs + pos + tid] >> shared.task[0].wbits : 0;
    if (tid < 32) shared.data[tid + partSize] = tid + partSize < dataLen ? samples[shared.task[0].samplesOffs + pos + tid + partSize] >> shared.task[0].wbits : 0;

    __syncthreads();

    shared.residual[tid] = 0;
    shared.coefs[tid] = threadIdx.x < shared.task[threadIdx.y].residualOrder ? tasks[blockIdx.y * blockDim.y + threadIdx.y].coefs[threadIdx.x] : 0;

    const int residualLen = max(0,min(shared.task[0].blocksize - pos - shared.task[threadIdx.y].residualOrder, partSize));
    for (int i = blockDim.y * (shared.task[threadIdx.y].type == Verbatim); i < blockDim.y; i++) // += 32
    {
	// compute residual
	int *co = &shared.coefs[threadIdx.y << 5];
	int ptr = threadIdx.x + (i << 5) + shared.task[threadIdx.y].residualOrder;
	int sum = 0;
	for (int c = -shared.task[threadIdx.y].residualOrder; c < 0; c++)
	    sum += __mul24(shared.data[ptr + c], *(co++));
	sum = shared.data[ptr] - (sum >> shared.task[threadIdx.y].shift);
	shared.residual[tid] += __mul24(ptr < dataLen, min(0x7fffff,(sum << 1) ^ (sum >> 31)));
    }

    shared.residual[tid] = shared.residual[tid] + shared.residual[tid + 8] + shared.residual[tid + 16] + shared.residual[tid + 24];
    shared.residual[tid] = shared.residual[tid] + shared.residual[tid + 2] + shared.residual[tid + 4] + shared.residual[tid + 6];
    if (threadIdx.x == 0)
	output[(blockIdx.y * blockDim.y + threadIdx.y) * 64 + blockIdx.x] = shared.residual[tid] + shared.residual[tid + 1];
}

extern "C" __global__ void cudaEstimateResidual1(
    int*output,
    int*samples,
    FlaCudaSubframeTask *tasks,
    int max_order,
    int partSize // should be blockDim.x * blockDim.y == 256
    )
{
    __shared__ struct {
	int data[32*9];
	volatile int residual[32*8];
	FlaCudaSubframeTask task;
	volatile int pos;
	volatile int dataLen;
    } shared;
    const int tid = threadIdx.x + threadIdx.y * 32;
    if (tid < sizeof(shared.task)/sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(&tasks[blockIdx.y]))[tid];
    if (tid == 0)
    {
	shared.pos = blockIdx.x * partSize; 
	shared.dataLen =  min(shared.task.data.blocksize - shared.pos, partSize + shared.task.data.residualOrder);
    }
    __syncthreads();

    // fetch samples
    shared.data[tid] = tid < shared.dataLen ? samples[shared.task.data.samplesOffs + shared.pos + tid] >> shared.task.data.wbits : 0;
    if (tid < 32) shared.data[tid + partSize] = tid + partSize < shared.dataLen ? samples[shared.task.data.samplesOffs + shared.pos + tid + partSize] >> shared.task.data.wbits : 0;
    __syncthreads();

    // compute residual
    int *co = &shared.task.coefs[0];
    int ptr = tid + shared.task.data.residualOrder;
    int sum = 0;
    for (int c = -shared.task.data.residualOrder; c < 0; c++)
	sum += __mul24(shared.data[ptr + c], *(co++));
    sum = shared.data[ptr] - (sum >> shared.task.data.shift);
    shared.residual[tid] = __mul24(ptr < shared.dataLen, min(0x7fffff,(sum << 1) ^ (sum >> 31)));
    __syncthreads();
    SUM256(shared.residual, tid, +=);
    if (tid == 0)
	output[blockIdx.y * 64 + blockIdx.x] = shared.residual[0];
}

extern "C" __global__ void cudaEstimateResidual8(
    int*output,
    int*samples,
    FlaCudaSubframeTask *tasks,
    int max_order,
    int partSize // should be blockDim.x * blockDim.y == 256
    )
{
    __shared__ struct {
	volatile int data[32*9];
	volatile int residual[32*8];
	FlaCudaSubframeData task[8];
	int coefs[32*8];
	volatile int pos;
	volatile int dataLen;
	volatile int dataOffs;
    } shared;
    const int tid = threadIdx.x + threadIdx.y * 32;
    const int taskNo = FASTMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    if (threadIdx.x < sizeof(FlaCudaSubframeData)/sizeof(int))
	((int*)&shared.task[threadIdx.y])[threadIdx.x] = ((int*)(&tasks[taskNo]))[threadIdx.x];
    const int ro = shared.task[threadIdx.y].residualOrder;
    shared.coefs[tid] = threadIdx.x < ro ? tasks[taskNo].coefs[threadIdx.x] : 0;
    if (tid == 0)
    {
	shared.pos = FASTMUL(blockIdx.x, partSize);
	shared.dataLen =  min(shared.task[0].blocksize - shared.pos, partSize + max_order);
	shared.dataOffs = shared.task[0].samplesOffs + shared.pos;
    }
    __syncthreads();

    // fetch samples
    if (tid < shared.dataLen)
	shared.data[tid] = samples[shared.dataOffs + tid] >> shared.task[0].wbits;
    if (tid + partSize < shared.dataLen)
	shared.data[tid + partSize] = samples[shared.dataOffs + tid + partSize] >> shared.task[0].wbits;

    __syncthreads();

    const int residualLen = max(0,min(shared.dataLen - ro, partSize));
    const int ptr2 = threadIdx.y << 5;
    int s = 0;
    for (int ptr = threadIdx.x; ptr < residualLen; ptr += 32)
    {
	// compute residual
	int sum = 
	    __mul24(shared.data[ptr + 0], shared.coefs[ptr2 + 0]) +
	    __mul24(shared.data[ptr + 1], shared.coefs[ptr2 + 1]) +
	    __mul24(shared.data[ptr + 2], shared.coefs[ptr2 + 2]) +
	    __mul24(shared.data[ptr + 3], shared.coefs[ptr2 + 3]);
	sum +=
	    __mul24(shared.data[ptr + 4], shared.coefs[ptr2 + 4]) +
	    __mul24(shared.data[ptr + 5], shared.coefs[ptr2 + 5]) +
	    __mul24(shared.data[ptr + 6], shared.coefs[ptr2 + 6]) +
	    __mul24(shared.data[ptr + 7], shared.coefs[ptr2 + 7]);
	sum = shared.data[ptr + ro] - (sum >> shared.task[threadIdx.y].shift);
	s += min(0x7fffff,(sum << 1) ^ (sum >> 31));
    }

    shared.residual[tid] = s;
    shared.residual[tid] = shared.residual[tid] + shared.residual[tid + 8] + shared.residual[tid + 16] + shared.residual[tid + 24];
    shared.residual[tid] = shared.residual[tid] + shared.residual[tid + 2] + shared.residual[tid + 4] + shared.residual[tid + 6];
    if (threadIdx.x == 0)
	output[(blockIdx.y * blockDim.y + threadIdx.y) * 64 + blockIdx.x] = shared.residual[tid] + shared.residual[tid + 1];
}

extern "C" __global__ void cudaEstimateResidual12(
    int*output,
    int*samples,
    FlaCudaSubframeTask *tasks,
    int max_order,
    int partSize // should be blockDim.x * blockDim.y == 256
    )
{
    __shared__ struct {
	volatile int data[32*9];
	volatile int residual[32*8];
	FlaCudaSubframeData task[8];
	int coefs[8*32];
	volatile int pos;
	volatile int dataLen;
	volatile int dataOffs;
    } shared;
    const int tid = threadIdx.x + threadIdx.y * 32;
    const int taskNo = FASTMUL(blockIdx.y, blockDim.y) + threadIdx.y;
    if (threadIdx.x < sizeof(FlaCudaSubframeData)/sizeof(int))
	((int*)&shared.task[threadIdx.y])[threadIdx.x] = ((int*)(&tasks[taskNo]))[threadIdx.x];
    const int ro = shared.task[threadIdx.y].residualOrder;
    shared.coefs[tid] = threadIdx.x < ro ? tasks[taskNo].coefs[threadIdx.x] : 0;
    if (tid == 0)
    {
	shared.pos = FASTMUL(blockIdx.x, partSize);
	shared.dataLen =  min(shared.task[0].blocksize - shared.pos, partSize + max_order);
	shared.dataOffs = shared.task[0].samplesOffs + shared.pos;
    }
    __syncthreads();

    // fetch samples
    if (tid < shared.dataLen)
	shared.data[tid] = samples[shared.dataOffs + tid] >> shared.task[0].wbits;
    if (tid + partSize < shared.dataLen)
	shared.data[tid + partSize] = samples[shared.dataOffs + tid + partSize] >> shared.task[0].wbits;

    __syncthreads();

    int residualLen = max(0,min(shared.dataLen - ro, partSize));
    const int ptr2 = threadIdx.y << 5;
    int s = 0;
    for (int ptr = threadIdx.x; ptr < residualLen; ptr += 32)
    {
	// compute residual
	int sum =
    	    FASTMUL(shared.data[ptr + 0], shared.coefs[ptr2 + 0]) +
	    FASTMUL(shared.data[ptr + 1], shared.coefs[ptr2 + 1]) +
	    FASTMUL(shared.data[ptr + 2], shared.coefs[ptr2 + 2]) +
	    FASTMUL(shared.data[ptr + 3], shared.coefs[ptr2 + 3]);
	sum += 
	    FASTMUL(shared.data[ptr + 4], shared.coefs[ptr2 + 4]) +
	    FASTMUL(shared.data[ptr + 5], shared.coefs[ptr2 + 5]) +
	    FASTMUL(shared.data[ptr + 6], shared.coefs[ptr2 + 6]) +
	    FASTMUL(shared.data[ptr + 7], shared.coefs[ptr2 + 7]);
	sum +=
	    FASTMUL(shared.data[ptr + 8], shared.coefs[ptr2 + 8]) +
	    FASTMUL(shared.data[ptr + 9], shared.coefs[ptr2 + 9]) +
	    FASTMUL(shared.data[ptr + 10], shared.coefs[ptr2 + 10]) +
	    FASTMUL(shared.data[ptr + 11], shared.coefs[ptr2 + 11]);
	sum = shared.data[ptr + ro] - (sum >> shared.task[threadIdx.y].shift);
	s += min(0x7fffff,(sum << 1) ^ (sum >> 31));
    }

    shared.residual[tid] = s;
    shared.residual[tid] = shared.residual[tid] + shared.residual[tid + 8] + shared.residual[tid + 16] + shared.residual[tid + 24];
    shared.residual[tid] = shared.residual[tid] + shared.residual[tid + 2] + shared.residual[tid + 4] + shared.residual[tid + 6];
    if (threadIdx.x == 0)
	output[(blockIdx.y * blockDim.y + threadIdx.y) * 64 + blockIdx.x] = shared.residual[tid] + shared.residual[tid + 1];
}

extern "C" __global__ void cudaChooseBestMethod(
    FlaCudaSubframeTask *tasks,
    int *residual,
    int partSize,
    int partCount, // <= blockDim.y (256)
    int taskCount
    )
{
    __shared__ struct {
	volatile int index[128];
	volatile int length[256];
	volatile int partLen[256];
	volatile FlaCudaSubframeTask task[8];
    } shared;
    const int tid = threadIdx.x + threadIdx.y * 32;
    
    shared.length[tid] = 0x7fffffff;
    for (int task = 0; task < taskCount; task += blockDim.y)
	if (task + threadIdx.y < taskCount)
	{
	    // fetch task data
	    ((int*)&shared.task[threadIdx.y])[threadIdx.x] = ((int*)(tasks + task + threadIdx.y + taskCount * blockIdx.y))[threadIdx.x];

	    int sum = 0;
	    for (int pos = threadIdx.x; pos < partCount; pos += blockDim.x)
	    {
		// fetch part sum
		int psum = residual[pos + 64 * (task + threadIdx.y + taskCount * blockIdx.y)];
		// calculate part size
		int residualLen = max(0,min(shared.task[threadIdx.y].data.blocksize - FASTMUL(pos, partSize) - shared.task[threadIdx.y].data.residualOrder, partSize));
		residualLen = FASTMUL(residualLen, shared.task[threadIdx.y].data.type != Constant || psum != 0);
		// calculate rice parameter
		int k = max(0, min(14, __float2int_rz(__log2f((psum + 0.000001f) / (residualLen + 0.000001f) + 0.5f))));
		// calculate part bit length
		sum += FASTMUL(residualLen, k + 1) + (psum >> k);
	    }
	    shared.partLen[tid] = sum;

	    // length sum: reduction in shared mem
	    shared.partLen[tid] += shared.partLen[tid + 16];
	    shared.partLen[tid] += shared.partLen[tid + 8];
	    shared.partLen[tid] += shared.partLen[tid + 4];
	    shared.partLen[tid] += shared.partLen[tid + 2];
	    shared.partLen[tid] += shared.partLen[tid + 1];
	    // return sum
	    if (threadIdx.x == 0)
	    {
		int obits = shared.task[threadIdx.y].data.obits - shared.task[threadIdx.y].data.wbits;
		shared.length[task + threadIdx.y] =
		    min(obits * shared.task[threadIdx.y].data.blocksize,
			shared.task[threadIdx.y].data.type == Fixed ? shared.task[threadIdx.y].data.residualOrder * obits + 6 + (4 * partCount/2) + shared.partLen[threadIdx.y * 32] :
			shared.task[threadIdx.y].data.type == LPC ? shared.task[threadIdx.y].data.residualOrder * obits + 4 + 5 + shared.task[threadIdx.y].data.residualOrder * shared.task[threadIdx.y].data.cbits + 6 + (4 * partCount/2)/* << porder */ + shared.partLen[threadIdx.y * 32] :
			shared.task[threadIdx.y].data.type == Constant ? obits * (1 + shared.task[threadIdx.y].data.blocksize * (shared.partLen[threadIdx.y * 32] != 0)) : 
			obits * shared.task[threadIdx.y].data.blocksize);
	    }
	}
    //shared.index[threadIdx.x] = threadIdx.x;
    //shared.length[threadIdx.x] = (threadIdx.x < taskCount) ? tasks[threadIdx.x + taskCount * blockIdx.y].size : 0x7fffffff;

    __syncthreads();

    if (tid < taskCount)
	tasks[tid + taskCount * blockIdx.y].data.size = shared.length[tid];

    __syncthreads();
    int l1 = shared.length[tid];
    if (tid < 128)
    {
	int l2 = shared.length[tid + 128];
	shared.index[tid] = tid + ((l2 < l1) << 7);
	shared.length[tid] = l1 = min(l1, l2);
    }
    __syncthreads();
    if (tid < 64)
    {
	int l2 = shared.length[tid + 64];
	shared.index[tid] = shared.index[tid + ((l2 < l1) << 6)];
	shared.length[tid] = l1 = min(l1, l2);
    }
    __syncthreads();
    if (tid < 32)
    {
#pragma unroll 5
	for (int sh = 5; sh > 0; sh --)
	{
	    int l2 = shared.length[tid + (1 << sh)];
	    shared.index[tid] = shared.index[tid + ((l2 < l1) << sh)];
	    shared.length[tid] = l1 = min(l1, l2);
	}
	if (tid == 0)
	    tasks[taskCount * blockIdx.y].data.best_index = taskCount * blockIdx.y + shared.index[shared.length[1] < shared.length[0]];
    }
}

extern "C" __global__ void cudaCopyBestMethod(
    FlaCudaSubframeTask *tasks_out,
    FlaCudaSubframeTask *tasks,
    int count
    )
{
    __shared__ struct {
	int best_index;
    } shared;
    if (threadIdx.x == 0)
	shared.best_index = tasks[count * blockIdx.y].data.best_index;
    __syncthreads();
    if (threadIdx.x < sizeof(FlaCudaSubframeTask)/sizeof(int))
	((int*)(tasks_out + blockIdx.y))[threadIdx.x] = ((int*)(tasks + shared.best_index))[threadIdx.x];
}

extern "C" __global__ void cudaCopyBestMethodStereo(
    FlaCudaSubframeTask *tasks_out,
    FlaCudaSubframeTask *tasks,
    int count
    )
{
    __shared__ struct {
	int best_index[4];
	int best_size[4];
	int lr_index[2];
    } shared;
    if (threadIdx.x < 4)
	shared.best_index[threadIdx.x] = tasks[count * (blockIdx.y * 4 + threadIdx.x)].data.best_index;
    if (threadIdx.x < 4)
	shared.best_size[threadIdx.x] = tasks[shared.best_index[threadIdx.x]].data.size;
    __syncthreads();
    if (threadIdx.x == 0)
    {
	int bitsBest = 0x7fffffff;
	if (bitsBest > shared.best_size[2] + shared.best_size[3]) // MidSide
	{
	    bitsBest = shared.best_size[2] + shared.best_size[3];
	    shared.lr_index[0] = shared.best_index[2];
	    shared.lr_index[1] = shared.best_index[3];
	}
	if (bitsBest > shared.best_size[3] + shared.best_size[1]) // RightSide
	{
	    bitsBest = shared.best_size[3] + shared.best_size[1];
	    shared.lr_index[0] = shared.best_index[3];
	    shared.lr_index[1] = shared.best_index[1];
	}
	if (bitsBest > shared.best_size[0] + shared.best_size[3]) // LeftSide
	{
	    bitsBest = shared.best_size[0] + shared.best_size[3];
	    shared.lr_index[0] = shared.best_index[0];
	    shared.lr_index[1] = shared.best_index[3];
	}
	if (bitsBest > shared.best_size[0] + shared.best_size[1]) // LeftRight
	{
	    bitsBest = shared.best_size[0] + shared.best_size[1];
	    shared.lr_index[0] = shared.best_index[0];
	    shared.lr_index[1] = shared.best_index[1];
	}
    }
    __syncthreads();
    if (threadIdx.x < sizeof(FlaCudaSubframeTask)/sizeof(int))
	((int*)(tasks_out + 2 * blockIdx.y))[threadIdx.x] = ((int*)(tasks + shared.lr_index[0]))[threadIdx.x];
    if (threadIdx.x == 0)
	tasks_out[2 * blockIdx.y].data.residualOffs = tasks[shared.best_index[0]].data.residualOffs;
    if (threadIdx.x < sizeof(FlaCudaSubframeTask)/sizeof(int))
	((int*)(tasks_out + 2 * blockIdx.y + 1))[threadIdx.x] = ((int*)(tasks + shared.lr_index[1]))[threadIdx.x];
    if (threadIdx.x == 0)
	tasks_out[2 * blockIdx.y + 1].data.residualOffs = tasks[shared.best_index[1]].data.residualOffs;
}

extern "C" __global__ void cudaEncodeResidual(
    int*output,
    int*samples,
    FlaCudaSubframeTask *tasks
    )
{
    __shared__ struct {
	int data[256 + 32];
	FlaCudaSubframeTask task;
    } shared;
    const int tid = threadIdx.x;
    if (threadIdx.x < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[threadIdx.x] = ((int*)(&tasks[blockIdx.y]))[threadIdx.x];
    __syncthreads();
    const int partSize = blockDim.x;
    const int pos = blockIdx.x * partSize;
    const int dataLen = min(shared.task.data.blocksize - pos, partSize + shared.task.data.residualOrder);

    // fetch samples
    shared.data[tid] = tid < dataLen ? samples[shared.task.data.samplesOffs + pos + tid] >> shared.task.data.wbits : 0;
    if (tid < 32) shared.data[tid + partSize] = tid + partSize < dataLen ? samples[shared.task.data.samplesOffs + pos + tid + partSize] >> shared.task.data.wbits : 0;
    const int residualLen = max(0,min(shared.task.data.blocksize - pos - shared.task.data.residualOrder, partSize));

    __syncthreads();    
    // compute residual
    int sum = 0;
    for (int c = 0; c < shared.task.data.residualOrder; c++)
	sum += __mul24(shared.data[tid + c], shared.task.coefs[c]);
    __syncthreads();
    shared.data[tid + shared.task.data.residualOrder] -= (sum >> shared.task.data.shift);
    __syncthreads();
    if (tid >= shared.task.data.residualOrder && tid < residualLen + shared.task.data.residualOrder)
	output[shared.task.data.residualOffs + pos + tid] = shared.data[tid];
    if (tid + 256 < residualLen + shared.task.data.residualOrder)
	output[shared.task.data.residualOffs + pos + tid + 256] = shared.data[tid + 256];
}

extern "C" __global__ void cudaCalcPartition(
    int* partition_lengths,
    int* residual,
    int* samples,
    FlaCudaSubframeTask *tasks,
    int max_porder, // <= 8
    int psize, // == (shared.task.data.blocksize >> max_porder), < 256
    int parts_per_block // == 256 / psize, > 0, <= 16
    )
{
    __shared__ struct {
	int data[256+32];
	FlaCudaSubframeTask task;
    } shared;
    const int tid = threadIdx.x + (threadIdx.y << 4);
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(&tasks[blockIdx.y]))[tid];
    __syncthreads();

    const int parts = min(parts_per_block, (1 << max_porder) - blockIdx.x * parts_per_block);
    const int offs = blockIdx.x * psize * parts_per_block + tid;

    // fetch samples
    if (tid < 32) shared.data[tid] = min(offs, tid + shared.task.data.residualOrder) >= 32 ? samples[shared.task.data.samplesOffs + offs - 32] >> shared.task.data.wbits : 0;
    shared.data[32 + tid] = tid < parts * psize ? samples[shared.task.data.samplesOffs + offs] >> shared.task.data.wbits : 0;
    __syncthreads();

    // compute residual
    int s = 0;
    for (int c = -shared.task.data.residualOrder; c < 0; c++)
	s += __mul24(shared.data[32 + tid + c], shared.task.coefs[shared.task.data.residualOrder + c]);
    s = shared.data[32 + tid] - (s >> shared.task.data.shift);

    if (offs >= shared.task.data.residualOrder && tid < parts * psize)
	residual[shared.task.data.residualOffs + offs] = s;
    else
	s = 0;

    // convert to unsigned
    s = min(0xfffff, (s << 1) ^ (s >> 31));

    //__syncthreads();
    //shared.data[tid] = s;
    //__syncthreads();

    //shared.data[tid] = (shared.data[tid] & (0x0000ffff << (tid & 16))) | (((shared.data[tid ^ 16] & (0x0000ffff << (tid & 16))) << (~tid & 16)) >> (tid & 16));
    //shared.data[tid] = (shared.data[tid] & (0x00ff00ff << (tid & 8))) | (((shared.data[tid ^ 8] & (0x00ff00ff << (tid & 8))) << (~tid & 8)) >> (tid & 8));
    //shared.data[tid] = (shared.data[tid] & (0x0f0f0f0f << (tid & 4))) | (((shared.data[tid ^ 4] & (0x0f0f0f0f << (tid & 4))) << (~tid & 4)) >> (tid & 4));
    //shared.data[tid] = (shared.data[tid] & (0x33333333 << (tid & 2))) | (((shared.data[tid ^ 2] & (0x33333333 << (tid & 2))) << (~tid & 2)) >> (tid & 2));
    //shared.data[tid] = (shared.data[tid] & (0x55555555 << (tid & 1))) | (((shared.data[tid ^ 1] & (0x55555555 << (tid & 1))) << (~tid & 1)) >> (tid & 1));
    //shared.data[tid] = __popc(shared.data[tid]);

    __syncthreads();
    shared.data[tid + (tid / psize)] = s;
    //shared.data[tid] = s;
    __syncthreads();

    s = (psize - shared.task.data.residualOrder * (threadIdx.x + blockIdx.x == 0)) * (threadIdx.y + 1);
    int dpos = __mul24(threadIdx.x, psize + 1);
    //int dpos = __mul24(threadIdx.x, psize);
    // calc number of unary bits for part threadIdx.x with rice paramater threadIdx.y
#pragma unroll 0
    for (int i = 0; i < psize; i++)
	s += shared.data[dpos + i] >> threadIdx.y;

    // output length
    const int pos = (15 << (max_porder + 1)) * blockIdx.y + (threadIdx.y << (max_porder + 1));
    if (threadIdx.y <= 14 && threadIdx.x < parts)
	partition_lengths[pos + blockIdx.x * parts_per_block + threadIdx.x] = s;
}

extern "C" __global__ void cudaCalcPartition16(
    int* partition_lengths,
    int* residual,
    int* samples,
    FlaCudaSubframeTask *tasks,
    int max_porder, // <= 8
    int psize, // == 16
    int parts_per_block // == 16
    )
{
    __shared__ struct {
	int data[256+32];
	FlaCudaSubframeTask task;
    } shared;
    const int tid = threadIdx.x + (threadIdx.y << 4);
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(&tasks[blockIdx.y]))[tid];
    __syncthreads();

    const int offs = (blockIdx.x << 8) + tid;

    // fetch samples
    if (tid < 32) shared.data[tid] = min(offs, tid + shared.task.data.residualOrder) >= 32 ? samples[shared.task.data.samplesOffs + offs - 32] >> shared.task.data.wbits : 0;
    shared.data[32 + tid] = samples[shared.task.data.samplesOffs + offs] >> shared.task.data.wbits;
 //   if (tid < 32 && tid >= shared.task.data.residualOrder)
	//shared.task.coefs[tid] = 0;
    __syncthreads();

    // compute residual
    int s = 0;
    for (int c = -shared.task.data.residualOrder; c < 0; c++)
	s += __mul24(shared.data[32 + tid + c], shared.task.coefs[shared.task.data.residualOrder + c]);
 //   int spos = 32 + tid - shared.task.data.residualOrder;
 //   int s=
	//__mul24(shared.data[spos + 0], shared.task.coefs[0]) + __mul24(shared.data[spos + 1], shared.task.coefs[1]) + 
	//__mul24(shared.data[spos + 2], shared.task.coefs[2]) + __mul24(shared.data[spos + 3], shared.task.coefs[3]) + 
	//__mul24(shared.data[spos + 4], shared.task.coefs[4]) + __mul24(shared.data[spos + 5], shared.task.coefs[5]) + 
	//__mul24(shared.data[spos + 6], shared.task.coefs[6]) + __mul24(shared.data[spos + 7], shared.task.coefs[7]) +
	//__mul24(shared.data[spos + 8], shared.task.coefs[8]) + __mul24(shared.data[spos + 9], shared.task.coefs[9]) + 
	//__mul24(shared.data[spos + 10], shared.task.coefs[10]) + __mul24(shared.data[spos + 11], shared.task.coefs[11]) +
	//__mul24(shared.data[spos + 12], shared.task.coefs[12]) + __mul24(shared.data[spos + 13], shared.task.coefs[13]) + 
	//__mul24(shared.data[spos + 14], shared.task.coefs[14]) + __mul24(shared.data[spos + 15], shared.task.coefs[15]);
    s = shared.data[32 + tid] - (s >> shared.task.data.shift);

    if (blockIdx.x != 0 || tid >= shared.task.data.residualOrder)
	residual[shared.task.data.residualOffs + (blockIdx.x << 8) + tid] = s;
    else
	s = 0;

    // convert to unsigned
    s = min(0xfffff, (s << 1) ^ (s >> 31));
    __syncthreads();
    shared.data[tid + threadIdx.y] = s;
    __syncthreads();

    // calc number of unary bits for part threadIdx.x with rice paramater threadIdx.y
    int dpos = __mul24(threadIdx.x, 17);
    int sum =
	(shared.data[dpos + 0] >> threadIdx.y) + (shared.data[dpos + 1] >> threadIdx.y) + 
	(shared.data[dpos + 2] >> threadIdx.y) + (shared.data[dpos + 3] >> threadIdx.y) + 
	(shared.data[dpos + 4] >> threadIdx.y) + (shared.data[dpos + 5] >> threadIdx.y) + 
	(shared.data[dpos + 6] >> threadIdx.y) + (shared.data[dpos + 7] >> threadIdx.y) + 
	(shared.data[dpos + 8] >> threadIdx.y) + (shared.data[dpos + 9] >> threadIdx.y) + 
	(shared.data[dpos + 10] >> threadIdx.y) + (shared.data[dpos + 11] >> threadIdx.y) + 
	(shared.data[dpos + 12] >> threadIdx.y) + (shared.data[dpos + 13] >> threadIdx.y) + 
	(shared.data[dpos + 14] >> threadIdx.y) + (shared.data[dpos + 15] >> threadIdx.y);

    // output length
    const int pos = ((15 * blockIdx.y + threadIdx.y) << (max_porder + 1)) + (blockIdx.x << 4) + threadIdx.x;
    if (threadIdx.y <= 14)
	partition_lengths[pos] = sum + (16 - shared.task.data.residualOrder * (threadIdx.x + blockIdx.x == 0)) * (threadIdx.y + 1);
}

extern "C" __global__ void cudaCalcLargePartition(
    int* partition_lengths,
    int* residual,
    int* samples,
    FlaCudaSubframeTask *tasks,
    int max_porder, // <= 8
    int psize, // == >= 128
    int parts_per_block // == 1
    )
{
    __shared__ struct {
	int data[256];
	volatile int length[256];
	FlaCudaSubframeTask task;
    } shared;
    const int tid = threadIdx.x + (threadIdx.y << 4);
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(&tasks[blockIdx.y]))[tid];
    __syncthreads();

    int sum = 0;
    for (int pos = 0; pos < psize; pos += 256)
    {
	// fetch residual
	int offs = blockIdx.x * psize + pos + tid;
	int s = (offs >= shared.task.data.residualOrder && pos + tid < psize) ? residual[shared.task.data.residualOffs + offs] : 0;
	// convert to unsigned
	shared.data[tid] = min(0xfffff, (s << 1) ^ (s >> 31));
	__syncthreads();

	// calc number of unary bits for each residual sample with each rice paramater
#pragma unroll 0
	for (int i = threadIdx.x; i < min(psize,256); i += 16)
	    // for sample (i + threadIdx.x) with this rice paramater (threadIdx.y)
	    sum += shared.data[i] >> threadIdx.y;
	__syncthreads();
    }
    shared.length[tid] = min(0xfffff,sum);
    SUM16(shared.length,tid,+=);

    // output length
    const int pos = (15 << (max_porder + 1)) * blockIdx.y + (threadIdx.y << (max_porder + 1));
    if (threadIdx.y <= 14 && threadIdx.x == 0)
	partition_lengths[pos + blockIdx.x] = min(0xfffff,shared.length[tid]) + (psize - shared.task.data.residualOrder * (blockIdx.x == 0)) * (threadIdx.y + 1);
}

// Sums partition lengths for a certain k == blockIdx.x
// Requires 128 threads
extern "C" __global__ void cudaSumPartition(
    int* partition_lengths,
    int max_porder
    )
{
    __shared__ struct {
	volatile int data[512+32]; // max_porder <= 8, data length <= 1 << 9.
    } shared;

    const int pos = (15 << (max_porder + 1)) * blockIdx.y + (blockIdx.x << (max_porder + 1));

    // fetch partition lengths
    shared.data[threadIdx.x] = threadIdx.x < (1 << max_porder) ? partition_lengths[pos + threadIdx.x] : 0;
    shared.data[blockDim.x + threadIdx.x] = blockDim.x + threadIdx.x < (1 << max_porder) ? partition_lengths[pos + blockDim.x + threadIdx.x] : 0;
    __syncthreads();

    int in_pos = (threadIdx.x << 1);
    int out_pos = (1 << max_porder) + threadIdx.x;
    int bs;
    for (bs = 1 << (max_porder - 1); bs > 32; bs >>= 1)
    {
	if (threadIdx.x < bs) shared.data[out_pos] = shared.data[in_pos] + shared.data[in_pos + 1];
	in_pos += bs << 1;
	out_pos += bs;
	__syncthreads();
    }
    if (threadIdx.x < 32)
    for (; bs > 0; bs >>= 1)
    {
	shared.data[out_pos] = shared.data[in_pos] + shared.data[in_pos + 1];
	in_pos += bs << 1;
	out_pos += bs;
    }
    __syncthreads();
    if (threadIdx.x < (1 << max_porder))
	partition_lengths[pos + (1 << max_porder) + threadIdx.x] = shared.data[(1 << max_porder) + threadIdx.x];
    if (blockDim.x + threadIdx.x < (1 << max_porder))
	partition_lengths[pos + (1 << max_porder) + blockDim.x + threadIdx.x] = shared.data[(1 << max_porder) + blockDim.x + threadIdx.x];
}

// Finds optimal rice parameter for up to 16 partitions at a time.
// Requires 16x16 threads
extern "C" __global__ void cudaFindRiceParameter(
    int* rice_parameters,
    int* partition_lengths,
    int max_porder
    )
{
    __shared__ struct {
	volatile int length[256];
	volatile int index[256];
    } shared;
    const int tid = threadIdx.x + (threadIdx.y << 5);
    const int parts = min(32, 2 << max_porder);
    const int pos = (15 << (max_porder + 1)) * blockIdx.y + (threadIdx.y << (max_porder + 1));

    // read length for 32 partitions
    int l1 = (threadIdx.x < parts) ? partition_lengths[pos + blockIdx.x * 32 + threadIdx.x] : 0xffffff;
    int l2 = (threadIdx.y + 8 <= 14 && threadIdx.x < parts) ? partition_lengths[pos + (8 << (max_porder + 1)) + blockIdx.x * 32 + threadIdx.x] : 0xffffff;
    // find best rice parameter
    shared.index[tid] = threadIdx.y + ((l2 < l1) << 3);
    shared.length[tid] = l1 = min(l1, l2);
    __syncthreads();
#pragma unroll 3
    for (int sh = 7; sh >= 5; sh --)
    {
	if (tid < (1 << sh))
	{
	    l2 = shared.length[tid + (1 << sh)];
	    shared.index[tid] = shared.index[tid + ((l2 < l1) << sh)];
	    shared.length[tid] = l1 = min(l1, l2);
	}    
	__syncthreads();
    }
    if (tid < parts)
    {
	// output rice parameter
	rice_parameters[(blockIdx.y << (max_porder + 2)) + blockIdx.x * parts + tid] = shared.index[tid];
	// output length
	rice_parameters[(blockIdx.y << (max_porder + 2)) + (1 << (max_porder + 1)) + blockIdx.x * parts + tid] = shared.length[tid];
    }
}

extern "C" __global__ void cudaFindPartitionOrder(
    int* best_rice_parameters,
    FlaCudaSubframeTask *tasks,
    int* rice_parameters,
    int max_porder
    )
{
    __shared__ struct {
	int data[512];
	volatile int tmp[256];
	int length[32];
	int index[32];
	//char4 ch[64];
	FlaCudaSubframeTask task;
    } shared;
    const int pos = (blockIdx.y << (max_porder + 2)) + (2 << max_porder);
    if (threadIdx.x < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[threadIdx.x] = ((int*)(&tasks[blockIdx.y]))[threadIdx.x];
    // fetch partition lengths
    shared.data[threadIdx.x] = threadIdx.x < (2 << max_porder) ? rice_parameters[pos + threadIdx.x] : 0;
    shared.data[threadIdx.x + 256] = threadIdx.x + 256 < (2 << max_porder) ? rice_parameters[pos + 256 + threadIdx.x] : 0;
    __syncthreads();

    for (int porder = max_porder; porder >= 0; porder--)
    {
	shared.tmp[threadIdx.x] = (threadIdx.x < (1 << porder)) * shared.data[(2 << max_porder) - (2 << porder) + threadIdx.x];
	__syncthreads();
	SUM256(shared.tmp, threadIdx.x, +=);
	if (threadIdx.x == 0)
	    shared.length[porder] = shared.tmp[0] + (4 << porder);
	__syncthreads();
    }

    if (threadIdx.x < 32)
    {
	shared.index[threadIdx.x] = threadIdx.x;
	if (threadIdx.x > max_porder)
	    shared.length[threadIdx.x] = 0xfffffff;
	int l1 = shared.length[threadIdx.x];
    #pragma unroll 4
	for (int sh = 3; sh >= 0; sh --)
	{
	    int l2 = shared.length[threadIdx.x + (1 << sh)];
	    shared.index[threadIdx.x] = shared.index[threadIdx.x + ((l2 < l1) << sh)];
	    shared.length[threadIdx.x] = l1 = min(l1, l2);
	}
	if (threadIdx.x == 0)
	    tasks[blockIdx.y].data.porder = shared.index[0];
	if (threadIdx.x == 0)
	{
	    int obits = shared.task.data.obits - shared.task.data.wbits;	    
	    tasks[blockIdx.y].data.size =
		shared.task.data.type == Fixed ? shared.task.data.residualOrder * obits + 6 + l1 :
		shared.task.data.type == LPC ? shared.task.data.residualOrder * obits + 6 + l1 + 4 + 5 + shared.task.data.residualOrder * shared.task.data.cbits :
		shared.task.data.type == Constant ? obits : obits * shared.task.data.blocksize;
	}
    }
    __syncthreads();
    int porder = shared.index[0];
    if (threadIdx.x < (1 << porder))
	best_rice_parameters[(blockIdx.y << max_porder) + threadIdx.x] = rice_parameters[pos - (2 << porder) + threadIdx.x];
    // FIXME: should be bytes?
 //   if (threadIdx.x < (1 << porder))
	//shared.tmp[threadIdx.x] = rice_parameters[pos - (2 << porder) + threadIdx.x];
 //   __syncthreads();
 //   if (threadIdx.x < max(1, (1 << porder) >> 2))
 //   {
	//char4 ch;
	//ch.x = shared.tmp[(threadIdx.x << 2)];
	//ch.y = shared.tmp[(threadIdx.x << 2) + 1];
	//ch.z = shared.tmp[(threadIdx.x << 2) + 2];
	//ch.w = shared.tmp[(threadIdx.x << 2) + 3];
	//shared.ch[threadIdx.x] = ch
 //   }	
 //   __syncthreads();
 //   if (threadIdx.x < max(1, (1 << porder) >> 2))
	//best_rice_parameters[(blockIdx.y << max_porder) + threadIdx.x] = shared.ch[threadIdx.x];
}

#endif

#if 0
    if (threadIdx.x < order)
    {
	for (int i = 0; i < order; i++)
	    if (threadIdx.x >= i)
		sum[threadIdx.x - i] += coefs[threadIdx.x] * sample[order - i - 1];
	fot (int i = order; i < blocksize; i++)
	{
	    if (!threadIdx.x) sample[order + i] = s = residual[order + i] + (sum[order + i] >> shift);
	    sum[threadIdx.x + i + 1] += coefs[threadIdx.x] * s;
	}
    }
#endif