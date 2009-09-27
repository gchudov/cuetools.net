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

typedef struct
{
    int samplesOffs;
    int windowOffs;
    int residualOffs;
    int blocksize;
    int reserved[12];
} computeAutocorTaskStruct;

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
    int reserved[4];
    int coefs[32];
} encodeResidualTaskStruct;

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
    encodeResidualTaskStruct *tasks,
    int *samples,
    int tasksPerChannel,
    int blocksize
)
{
    __shared__ struct {
	volatile int wbits[256];
	encodeResidualTaskStruct task;
    } shared;

    if (threadIdx.x < 16)
	((int*)&shared.task)[threadIdx.x] = ((int*)(&tasks[blockIdx.x * tasksPerChannel]))[threadIdx.x];
    shared.wbits[threadIdx.x] = 0;
    __syncthreads();

    for (int pos = 0; pos < blocksize; pos += blockDim.x)
	shared.wbits[threadIdx.x] |= pos + threadIdx.x < blocksize ? samples[shared.task.samplesOffs + pos + threadIdx.x] : 0;
    __syncthreads();

    if (threadIdx.x < 128) shared.wbits[threadIdx.x] |= shared.wbits[threadIdx.x + 128]; __syncthreads();
    if (threadIdx.x < 64) shared.wbits[threadIdx.x] |= shared.wbits[threadIdx.x + 64]; __syncthreads();
    if (threadIdx.x < 32) shared.wbits[threadIdx.x] |= shared.wbits[threadIdx.x + 32]; __syncthreads();
    shared.wbits[threadIdx.x] |= shared.wbits[threadIdx.x + 16];
    shared.wbits[threadIdx.x] |= shared.wbits[threadIdx.x + 8];
    shared.wbits[threadIdx.x] |= shared.wbits[threadIdx.x + 4];
    shared.wbits[threadIdx.x] |= shared.wbits[threadIdx.x + 2];
    shared.wbits[threadIdx.x] |= shared.wbits[threadIdx.x + 1];
    if (threadIdx.x < tasksPerChannel)
	tasks[blockIdx.x * tasksPerChannel + threadIdx.x].wbits = max(0,__ffs(shared.wbits[0]) - 1);
}

extern "C" __global__ void cudaComputeAutocor(
    float *output,
    const int *samples,
    const float *window,
    computeAutocorTaskStruct *tasks,
    int max_order, // should be <= 32
    int frameSize,
    int partSize // should be <= 2*blockDim - max_order
)
{
    __shared__ struct {
	float data[512];
	volatile float product[256];
	computeAutocorTaskStruct task;
    } shared;
    const int tid = threadIdx.x + (threadIdx.y * 32);
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    __syncthreads();

    // fetch samples
    {
	const int pos = blockIdx.x * partSize;
	const int dataLen = min(frameSize - pos, partSize + max_order);

	shared.data[tid] = tid < dataLen ? samples[shared.task.samplesOffs + pos + tid] * window[shared.task.windowOffs + pos + tid]: 0.0f;
	shared.data[tid + 256] = tid + 256 < dataLen ? samples[shared.task.samplesOffs + pos + tid + 256] * window[shared.task.windowOffs + pos + tid + 256]: 0.0f;
    }
    __syncthreads();

    for (int lag = threadIdx.y; lag <= max_order; lag += 8)
    {
        const int productLen = min(frameSize - blockIdx.x * partSize - lag, partSize);
	shared.product[tid] = 0.0;
	for (int ptr = threadIdx.x; ptr < productLen + threadIdx.x; ptr += 128)
	    shared.product[tid] += ((ptr < productLen) * shared.data[ptr] * shared.data[ptr + lag]
				 + (ptr + 32 < productLen) * shared.data[ptr + 32] * shared.data[ptr + 32 + lag])
				 + ((ptr + 64 < productLen) * shared.data[ptr + 64] * shared.data[ptr + 64 + lag]
				 + (ptr + 96 < productLen) * shared.data[ptr + 96] * shared.data[ptr + 96 + lag]);
	// product sum: reduction in shared mem
	//shared.product[tid] += shared.product[tid + 16];
	shared.product[tid] = (shared.product[tid] + shared.product[tid + 16]) + (shared.product[tid + 8] + shared.product[tid + 24]);
	shared.product[tid] = (shared.product[tid] + shared.product[tid + 4]) + (shared.product[tid + 2] + shared.product[tid + 6]);
	// return results
	if (threadIdx.x == 0)
	    output[(blockIdx.x + blockIdx.y * gridDim.x) * (max_order + 1) + lag] = shared.product[tid] + shared.product[tid + 1];
    }
}

extern "C" __global__ void cudaComputeLPC(
    encodeResidualTaskStruct *output,
    float*autoc,
    computeAutocorTaskStruct *tasks,
    int max_order, // should be <= 32
    int partCount // should be <= blockDim?
)
{
    __shared__ struct {
	computeAutocorTaskStruct task;
	volatile float ldr[32];
	volatile int   bits[32];
	volatile float autoc[33];
	volatile float gen0[32];
	volatile float gen1[32];
	volatile float parts[128];
	//volatile float reff[32];
	//int   cbits;
    } shared;
    const int tid = threadIdx.x;
    
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    
    // add up parts
    for (int order = 0; order <= max_order; order++)
    {
	shared.parts[tid] = tid < partCount ? autoc[(blockIdx.y * partCount + tid) * (max_order + 1) + order] : 0;
	__syncthreads();
	if (tid < 64 && blockDim.x > 64) shared.parts[tid] += shared.parts[tid + 64];
	__syncthreads();
	if (tid < 32) 
	{
	    if (blockDim.x > 32) shared.parts[tid] += shared.parts[tid + 32];
	    shared.parts[tid] += shared.parts[tid + 16];
	    shared.parts[tid] += shared.parts[tid + 8];
	    shared.parts[tid] += shared.parts[tid + 4];
	    shared.parts[tid] += shared.parts[tid + 2];
	    shared.parts[tid] += shared.parts[tid + 1];
	    if (tid == 0)
		shared.autoc[order] = shared.parts[0];
	}
    }
   
    if (tid < 32)
    {
	shared.gen0[tid] = shared.autoc[tid+1];
	shared.gen1[tid] = shared.autoc[tid+1];
	shared.ldr[tid] = 0.0f;

	float error = shared.autoc[0];
	for (int order = 0; order < max_order; order++)
	{
	    // Schur recursion
	    float reff = -shared.gen1[0] / error;
	    //if (tid == 0) shared.reff[order] = reff;
	    error += __fmul_rz(shared.gen1[0], reff);
	    if (tid < max_order - 1 - order)
	    {
		float g1 = shared.gen1[tid + 1] + __fmul_rz(reff, shared.gen0[tid]);
		float g0 = __fmul_rz(shared.gen1[tid + 1], reff) + shared.gen0[tid];
		shared.gen1[tid] = g1;
		shared.gen0[tid] = g0;
	    }

	    // Levinson-Durbin recursion
	    shared.ldr[tid] += (tid < order) * __fmul_rz(reff, shared.ldr[order - 1 - tid]) + (tid  == order) * reff;

	    // Quantization
	    int precision = 13 - (order > 8);
	    int taskNo = shared.task.residualOffs + order;
	    shared.bits[tid] = __mul24((33 - __clz(__float2int_rn(fabs(shared.ldr[tid]) * (1 << 15))) - precision), tid <= order);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 16]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 8]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 4]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 2]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 1]);
	    int sh = max(0,min(15, 15 - shared.bits[0]));
	    
	    // reverse coefs
	    int coef = max(-(1 << precision),min((1 << precision)-1,__float2int_rn(-shared.ldr[order - tid] * (1 << sh))));
	    if (tid <= order)
		output[taskNo].coefs[tid] = coef;
	    if (tid == 0)
		output[taskNo].shift = sh;
	    shared.bits[tid] = 33 - max(__clz(coef),__clz(-1 ^ coef));
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 16]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 8]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 4]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 2]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 1]);
	    int cbits = shared.bits[0];
	    if (tid == 0)
		output[taskNo].cbits = cbits;
	}
    }
}

#define SUM32(buf,tid)      buf[tid] += buf[tid + 16]; buf[tid] += buf[tid + 8]; buf[tid] += buf[tid + 4]; buf[tid] += buf[tid + 2]; buf[tid] += buf[tid + 1];
#define SUM64(buf,tid)      if (tid < 32) buf[tid] += buf[tid + 32]; __syncthreads(); if (tid < 32) SUM32(buf,tid)
#define SUM128(buf,tid)     if (tid < 64) buf[tid] += buf[tid + 64]; __syncthreads(); SUM64(buf,tid)
#define SUM256(buf,tid)     if (tid < 128) buf[tid] += buf[tid + 128]; __syncthreads(); SUM128(buf,tid)
#define SUM512(buf,tid)     if (tid < 256) buf[tid] += buf[tid + 256]; __syncthreads(); SUM256(buf,tid)

#define FSQR(s) ((s)*(s))

extern "C" __global__ void cudaComputeLPCLattice(
    encodeResidualTaskStruct *tasks,
    const int taskCount, // tasks per block
    const int *samples,
    const int frameSize, // <= 512
    const int max_order // should be <= 32
)
{
    __shared__ struct {
	encodeResidualTaskStruct task;
	volatile float F[512];
	volatile float B[512];
	volatile float tmp[256];
	volatile float arp[32];
	volatile float rc[32];
	volatile int   bits[32];
	volatile float PE[33];
	volatile float DEN, reff;
    } shared;

    // fetch task data
    if (threadIdx.x < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[threadIdx.x] = ((int*)(tasks + taskCount * blockIdx.y))[threadIdx.x];    
    __syncthreads();

    // F = samples; B = samples
    shared.F[threadIdx.x] = threadIdx.x < frameSize ? samples[shared.task.samplesOffs + threadIdx.x] >> shared.task.wbits : 0.0f;
    shared.F[threadIdx.x + 256] = threadIdx.x + 256 < frameSize ? samples[shared.task.samplesOffs + threadIdx.x + 256] >> shared.task.wbits : 0.0f;
    shared.B[threadIdx.x] = shared.F[threadIdx.x];
    shared.B[threadIdx.x + 256] = shared.F[threadIdx.x + 256];
    __syncthreads();

    // DEN = F*F'
    shared.tmp[threadIdx.x] = FSQR(shared.F[threadIdx.x]) + FSQR(shared.F[threadIdx.x + 256]);
    __syncthreads();
    SUM256(shared.tmp,threadIdx.x);
    if (threadIdx.x == 0)
    {
	shared.DEN = shared.tmp[0];
	shared.PE[0] = shared.tmp[0] / frameSize;
    }
    __syncthreads();

    for (int order = 1; order <= max_order; order++)
    {
	// reff = F(order+1:frameSize) * B(1:frameSize-order)' / DEN
	float f1 = (threadIdx.x + order < frameSize) * shared.F[order + threadIdx.x];
	float f2 = (threadIdx.x + 256 + order < frameSize) * shared.F[order + threadIdx.x + 256];
	shared.tmp[threadIdx.x] = f1 * shared.B[threadIdx.x] + f2 * shared.B[threadIdx.x + 256]; 
	__syncthreads(); 
	SUM256(shared.tmp, threadIdx.x);
	if (threadIdx.x == 0)
	    shared.reff = shared.tmp[0] / shared.DEN;
	__syncthreads();

	// arp(order) = rc(order) = reff
	if (threadIdx.x == 0)
	    shared.arp[order - 1] = shared.rc[order - 1] = shared.reff;

	// Levinson-Durbin recursion
	// arp(1:order-1) = arp(1:order-1) - reff * arp(order-1:-1:1)
	if (threadIdx.x < 32)
	    shared.arp[threadIdx.x] -= (threadIdx.x < order - 1) * __fmul_rz(shared.reff, shared.arp[order - 2 - threadIdx.x]);

	// F1 = F(order+1:frameSize) - reff * B(1:frameSize-order)
	// B(1:frameSize-order) = B(1:frameSize-order) - reff * F(order+1:frameSize)
	// F(order+1:frameSize) = F1
	if (threadIdx.x < frameSize - order)
	{
	    shared.F[order + threadIdx.x] -= shared.reff * shared.B[threadIdx.x];
	    shared.B[threadIdx.x] -= shared.reff * f1;
	}
	if (threadIdx.x + 256 < frameSize - order)
	{
	    shared.F[order + threadIdx.x + 256] -= shared.reff * shared.B[threadIdx.x + 256];
	    shared.B[threadIdx.x + 256] -= shared.reff * f2;
	}
	__syncthreads();

	// DEN = F(order+1:frameSize) * F(order+1:frameSize)' + B(1:frameSize-order) * B(1:frameSize-order)' (BURG)
	shared.tmp[threadIdx.x] = (threadIdx.x < frameSize - order) * (FSQR(shared.F[threadIdx.x + order]) + FSQR(shared.B[threadIdx.x]))
	    + (threadIdx.x + 256 < frameSize - order) * (FSQR(shared.F[threadIdx.x + 256 + order]) + FSQR(shared.B[threadIdx.x + 256]));
	__syncthreads();
	SUM256(shared.tmp, threadIdx.x);
	if (threadIdx.x == 0)
	{
	    shared.DEN = shared.tmp[0] / 2;
	    shared.PE[order] = shared.tmp[0] / 2 / (frameSize - order);
	}
	__syncthreads();

	// Quantization
	if (threadIdx.x < 32)
	{
	    int precision = 10 - (order > 8) - min(2, shared.task.wbits);
	    int taskNo = taskCount * blockIdx.y + order - 1;
	    shared.bits[threadIdx.x] = __mul24((33 - __clz(__float2int_rn(fabs(shared.arp[threadIdx.x]) * (1 << 15))) - precision), threadIdx.x < order);
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 16]);
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 8]);
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 4]);
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 2]);
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 1]);
	    int sh = max(0,min(15, 15 - shared.bits[0]));
            
	    // reverse coefs
	    int coef = max(-(1 << precision),min((1 << precision)-1,__float2int_rn(shared.arp[order - 1 - threadIdx.x] * (1 << sh))));
	    if (threadIdx.x < order)
		tasks[taskNo].coefs[threadIdx.x] = coef;
	    if (threadIdx.x == 0)
		tasks[taskNo].shift = sh;
	    shared.bits[threadIdx.x] = 33 - max(__clz(coef),__clz(-1 ^ coef));
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 16]);
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 8]);
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 4]);
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 2]);
	    shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 1]);
	    int cbits = shared.bits[0];
	    if (threadIdx.x == 0)
		tasks[taskNo].cbits = cbits;
	}
    }
}

extern "C" __global__ void cudaComputeLPCLattice512(
    encodeResidualTaskStruct *tasks,
    const int taskCount, // tasks per block
    const int *samples,
    const int frameSize, // <= 512
    const int max_order // should be <= 32
)
{
    __shared__ struct {
	encodeResidualTaskStruct task;
	float F[512];
	float B[512];
	float lpc[32][32];
	volatile float tmp[512];
	volatile float arp[32];
	volatile float rc[32];
	volatile int   bits[512];
	volatile float f, b;
    } shared;

    // fetch task data
    if (threadIdx.x < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[threadIdx.x] = ((int*)(tasks + taskCount * blockIdx.y))[threadIdx.x];    
    __syncthreads();

    // F = samples; B = samples
    shared.F[threadIdx.x] = threadIdx.x < frameSize ? samples[shared.task.samplesOffs + threadIdx.x] >> shared.task.wbits : 0.0f;
    shared.B[threadIdx.x] = shared.F[threadIdx.x];
    __syncthreads();

    // DEN = F*F'
    shared.tmp[threadIdx.x] = FSQR(shared.F[threadIdx.x]);
    __syncthreads();
    SUM512(shared.tmp,threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0)
	shared.f = shared.b = shared.tmp[0];
 //   if (threadIdx.x == 0)
	//shared.PE[0] = DEN / frameSize;
    __syncthreads();

    for (int order = 1; order <= max_order; order++)
    {
	// reff = F(order+1:frameSize) * B(1:frameSize-order)' / DEN
	shared.tmp[threadIdx.x] = (threadIdx.x + order < frameSize) * shared.F[threadIdx.x + order] * shared.B[threadIdx.x];
	__syncthreads(); 
	SUM512(shared.tmp, threadIdx.x);
	__syncthreads();
	
	//float reff = shared.tmp[0] * rsqrtf(shared.b * shared.f); // Geometric lattice
	float reff = shared.tmp[0] * 2 / (shared.b + shared.f); // Burg method
	__syncthreads();

	// Levinson-Durbin recursion
	// arp(order) = rc(order) = reff
	// arp(1:order-1) = arp(1:order-1) - reff * arp(order-1:-1:1)
	if (threadIdx.x == 32)
	    shared.arp[order - 1] = shared.rc[order - 1] = reff;
	if (threadIdx.x < 32)
	    shared.arp[threadIdx.x] -= (threadIdx.x < order - 1) * __fmul_rz(reff, shared.arp[order - 2 - threadIdx.x]);

	// F1 = F(order+1:frameSize) - reff * B(1:frameSize-order)
	// B(1:frameSize-order) = B(1:frameSize-order) - reff * F(order+1:frameSize)
	// F(order+1:frameSize) = F1
	if (threadIdx.x < frameSize - order)
	{
	    float f;// = shared.F[threadIdx.x + order];
	    shared.F[threadIdx.x + order] = (f = shared.F[threadIdx.x + order]) - reff * shared.B[threadIdx.x];
	    shared.B[threadIdx.x] -= reff * f;
	}
	__syncthreads();

	// f = F(order+1:frameSize) * F(order+1:frameSize)'
	// b = B(1:frameSize-order) * B(1:frameSize-order)'
	shared.tmp[threadIdx.x] = (threadIdx.x < frameSize - order) * FSQR(shared.F[threadIdx.x + order]);
	__syncthreads();
	SUM512(shared.tmp, threadIdx.x);
	__syncthreads();
	if (threadIdx.x == 0)
	    shared.f = shared.tmp[0];
	__syncthreads();

	shared.tmp[threadIdx.x] = (threadIdx.x < frameSize - order) * FSQR(shared.B[threadIdx.x]);
	__syncthreads();
	SUM512(shared.tmp, threadIdx.x);
	__syncthreads();
	if (threadIdx.x == 0)
	    shared.b = shared.tmp[0];
	__syncthreads();

	if (threadIdx.x < 32)
	    shared.lpc[order - 1][threadIdx.x] = shared.arp[threadIdx.x];

	//if (threadIdx.x == 0)
	//    shared.PE[order] = (shared.b + shared.f) / 2 / (frameSize - order);
	__syncthreads();
    }
    for (int order = 1 + (threadIdx.x >> 5); order <= max_order; order += 16)
    {
	// Quantization
	int cn = threadIdx.x & 31;
	int precision = 10 - (order > 8) - min(2, shared.task.wbits);
	int taskNo = taskCount * blockIdx.y + order - 1;
	shared.bits[threadIdx.x] = __mul24((33 - __clz(__float2int_rn(fabs(shared.lpc[order - 1][cn]) * (1 << 15))) - precision), cn < order);
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 16]);
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 8]);
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 4]);
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 2]);
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 1]);
	int sh = max(0,min(15, 15 - shared.bits[threadIdx.x - cn]));
            
	// reverse coefs
	int coef = max(-(1 << precision),min((1 << precision)-1,__float2int_rn(shared.lpc[order - 1][order - 1 - cn] * (1 << sh))));
	if (cn < order)
	    tasks[taskNo].coefs[cn] = coef;
	if (cn == 0)
	    tasks[taskNo].shift = sh;
	shared.bits[threadIdx.x] = 33 - max(__clz(coef),__clz(-1 ^ coef));
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 16]);
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 8]);
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 4]);
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 2]);
	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 1]);
	int cbits = shared.bits[threadIdx.x - cn];
	if (cn == 0)
	    tasks[taskNo].cbits = cbits;
    }
}

// blockDim.x == 32
// blockDim.y == 8
extern "C" __global__ void cudaEstimateResidual(
    int*output,
    int*samples,
    encodeResidualTaskStruct *tasks,
    int max_order,
    int frameSize,
    int partSize // should be blockDim.x * blockDim.y == 256
    )
{
    __shared__ struct {
	int data[32*9];
	volatile int residual[32*8];
	encodeResidualTaskStruct task[8];
    } shared;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (threadIdx.x < 16)
	((int*)&shared.task[threadIdx.y])[threadIdx.x] = ((int*)(&tasks[blockIdx.y * blockDim.y + threadIdx.y]))[threadIdx.x];
    __syncthreads();
    const int pos = blockIdx.x * partSize;
    const int dataLen = min(frameSize - pos, partSize + max_order);

    // fetch samples
    shared.data[tid] = tid < dataLen ? samples[shared.task[0].samplesOffs + pos + tid] >> shared.task[0].wbits : 0;
    if (tid < 32) shared.data[tid + partSize] = tid + partSize < dataLen ? samples[shared.task[0].samplesOffs + pos + tid + partSize] >> shared.task[0].wbits : 0;
    const int residualLen = max(0,min(frameSize - pos - shared.task[threadIdx.y].residualOrder, partSize));

    __syncthreads();

    shared.residual[tid] = 0;
    shared.task[threadIdx.y].coefs[threadIdx.x] = threadIdx.x < max_order ? tasks[blockIdx.y * blockDim.y + threadIdx.y].coefs[threadIdx.x] : 0;

    for (int i = blockDim.y * (shared.task[threadIdx.y].type == Verbatim); i < blockDim.y; i++) // += 32
    {
	int ptr = threadIdx.x + (i<<5);
	// compute residual
	int sum = 0;
	int c = 0;
	for (c = 0; c < shared.task[threadIdx.y].residualOrder; c++)
	    sum += __mul24(shared.data[ptr + c], shared.task[threadIdx.y].coefs[c]);
	sum = shared.data[ptr + c] - (sum >> shared.task[threadIdx.y].shift);
	shared.residual[tid] += __mul24(ptr < residualLen, min(0x7fffff,(sum << 1) ^ (sum >> 31)));
    }

    // enable this line when using blockDim.x == 64
    //__syncthreads(); if (threadIdx.x < 32) shared.residual[tid] += shared.residual[tid + 32]; __syncthreads();
    shared.residual[tid] += shared.residual[tid + 16];
    shared.residual[tid] += shared.residual[tid + 8];
    shared.residual[tid] += shared.residual[tid + 4];
    shared.residual[tid] += shared.residual[tid + 2];
    shared.residual[tid] += shared.residual[tid + 1];

    // rice parameter search
    shared.residual[tid] = (shared.task[threadIdx.y].type != Constant || shared.residual[threadIdx.y * blockDim.x] != 0) *
	(__mul24(threadIdx.x >= 15, 0x7fffff) + residualLen * (threadIdx.x + 1) + ((shared.residual[threadIdx.y * blockDim.x] - (residualLen >> 1)) >> threadIdx.x));
    shared.residual[tid] = min(shared.residual[tid], shared.residual[tid + 8]);
    shared.residual[tid] = min(shared.residual[tid], shared.residual[tid + 4]);
    shared.residual[tid] = min(shared.residual[tid], shared.residual[tid + 2]);
    shared.residual[tid] = min(shared.residual[tid], shared.residual[tid + 1]);
    if (threadIdx.x == 0)
	output[(blockIdx.y * blockDim.y + threadIdx.y) * 64 + blockIdx.x] = shared.residual[tid];
}

#define BEST_INDEX(a,b) ((a) + ((b) - (a)) * (shared.length[b] < shared.length[a]))

extern "C" __global__ void cudaChooseBestMethod(
    encodeResidualTaskStruct *tasks,
    int *residual,
    int partCount, // <= blockDim.y (256)
    int taskCount
    )
{
    __shared__ struct {
	volatile int index[128];
	volatile int partLen[512];
	int length[256];
	volatile encodeResidualTaskStruct task[16];
    } shared;
    const int tid = threadIdx.x + threadIdx.y * 32;
    
    if (tid < 256) shared.length[tid] = 0x7fffffff;
    for (int task = 0; task < taskCount; task += blockDim.y)
	if (task + threadIdx.y < taskCount)
	{
	    // fetch task data
	    ((int*)&shared.task[threadIdx.y])[threadIdx.x] = ((int*)(tasks + task + threadIdx.y + taskCount * blockIdx.y))[threadIdx.x];

	    int sum = 0;
	    for (int pos = 0; pos < partCount; pos += blockDim.x)
		sum += (pos + threadIdx.x < partCount ? residual[pos + threadIdx.x + 64 * (task + threadIdx.y + taskCount * blockIdx.y)] : 0);
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
		int obits = shared.task[threadIdx.y].obits - shared.task[threadIdx.y].wbits;
		shared.length[task + threadIdx.y] =
		    min(obits * shared.task[threadIdx.y].blocksize,
			shared.task[threadIdx.y].type == Fixed ? shared.task[threadIdx.y].residualOrder * obits + 6 + shared.partLen[threadIdx.y * 32] :
			shared.task[threadIdx.y].type == LPC ? shared.task[threadIdx.y].residualOrder * obits + 4 + 5 + shared.task[threadIdx.y].residualOrder * shared.task[threadIdx.y].cbits + 6 + (4 * partCount/2)/* << porder */ + shared.partLen[threadIdx.y * 32] :
			shared.task[threadIdx.y].type == Constant ? obits * (1 + shared.task[threadIdx.y].blocksize * (shared.partLen[threadIdx.y * 32] != 0)) : 
			obits * shared.task[threadIdx.y].blocksize);
	    }
	}
    //shared.index[threadIdx.x] = threadIdx.x;
    //shared.length[threadIdx.x] = (threadIdx.x < taskCount) ? tasks[threadIdx.x + taskCount * blockIdx.y].size : 0x7fffffff;

    __syncthreads();

    //if (tid < 128) shared.index[tid] = BEST_INDEX(shared.index[tid], shared.index[tid + 128]); __syncthreads();
    if (tid < 128) shared.index[tid] = BEST_INDEX(tid, tid + 128); __syncthreads();
    if (tid < 64) shared.index[tid] = BEST_INDEX(shared.index[tid], shared.index[tid + 64]); __syncthreads();
    if (tid < 32) 
    {
	shared.index[tid] = BEST_INDEX(shared.index[tid], shared.index[tid + 32]);
	shared.index[tid] = BEST_INDEX(shared.index[tid], shared.index[tid + 16]);
	shared.index[tid] = BEST_INDEX(shared.index[tid], shared.index[tid + 8]);
	shared.index[tid] = BEST_INDEX(shared.index[tid], shared.index[tid + 4]);
	shared.index[tid] = BEST_INDEX(shared.index[tid], shared.index[tid + 2]);
	shared.index[tid] = BEST_INDEX(shared.index[tid], shared.index[tid + 1]);
    }
    __syncthreads();
 //   if (threadIdx.x < sizeof(encodeResidualTaskStruct)/sizeof(int))
	//((int*)(tasks_out + blockIdx.y))[threadIdx.x] = ((int*)(tasks + taskCount * blockIdx.y + shared.index[0]))[threadIdx.x];
    if (tid == 0)
	tasks[taskCount * blockIdx.y].best_index = taskCount * blockIdx.y + shared.index[0];
    if (tid < taskCount)
	tasks[tid + taskCount * blockIdx.y].size = shared.length[tid];
}

extern "C" __global__ void cudaCopyBestMethod(
    encodeResidualTaskStruct *tasks_out,
    encodeResidualTaskStruct *tasks,
    int count
    )
{
    __shared__ struct {
	int best_index;
    } shared;
    if (threadIdx.x == 0)
	shared.best_index = tasks[count * blockIdx.y].best_index;
    __syncthreads();
    if (threadIdx.x < sizeof(encodeResidualTaskStruct)/sizeof(int))
	((int*)(tasks_out + blockIdx.y))[threadIdx.x] = ((int*)(tasks + shared.best_index))[threadIdx.x];
}

extern "C" __global__ void cudaCopyBestMethodStereo(
    encodeResidualTaskStruct *tasks_out,
    encodeResidualTaskStruct *tasks,
    int count
    )
{
    __shared__ struct {
	int best_index[4];
	int best_size[4];
	int lr_index[2];
    } shared;
    if (threadIdx.x < 4)
	shared.best_index[threadIdx.x] = tasks[count * (blockIdx.y * 4 + threadIdx.x)].best_index;
    if (threadIdx.x < 4)
	shared.best_size[threadIdx.x] = tasks[shared.best_index[threadIdx.x]].size;
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
    if (threadIdx.x < sizeof(encodeResidualTaskStruct)/sizeof(int))
	((int*)(tasks_out + 2 * blockIdx.y))[threadIdx.x] = ((int*)(tasks + shared.lr_index[0]))[threadIdx.x];
    if (threadIdx.x == 0)
	tasks_out[2 * blockIdx.y].residualOffs = tasks[shared.best_index[0]].residualOffs;
    if (threadIdx.x < sizeof(encodeResidualTaskStruct)/sizeof(int))
	((int*)(tasks_out + 2 * blockIdx.y + 1))[threadIdx.x] = ((int*)(tasks + shared.lr_index[1]))[threadIdx.x];
    if (threadIdx.x == 0)
	tasks_out[2 * blockIdx.y + 1].residualOffs = tasks[shared.best_index[1]].residualOffs;
}

extern "C" __global__ void cudaEncodeResidual(
    int*output,
    int*samples,
    encodeResidualTaskStruct *tasks
    )
{
    __shared__ struct {
	int data[256 + 32];
	encodeResidualTaskStruct task;
    } shared;
    const int tid = threadIdx.x;
    if (threadIdx.x < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[threadIdx.x] = ((int*)(&tasks[blockIdx.y]))[threadIdx.x];
    __syncthreads();
    const int partSize = blockDim.x;
    const int pos = blockIdx.x * partSize;
    const int dataLen = min(shared.task.blocksize - pos, partSize + shared.task.residualOrder);

    // fetch samples
    shared.data[tid] = tid < dataLen ? samples[shared.task.samplesOffs + pos + tid] >> shared.task.wbits : 0;
    if (tid < 32) shared.data[tid + partSize] = tid + partSize < dataLen ? samples[shared.task.samplesOffs + pos + tid + partSize] >> shared.task.wbits : 0;
    const int residualLen = max(0,min(shared.task.blocksize - pos - shared.task.residualOrder, partSize));

    __syncthreads();
    
    // compute residual
    int sum = 0;
    for (int c = 0; c < shared.task.residualOrder; c++)
	sum += __mul24(shared.data[tid + c], shared.task.coefs[c]);
    if (tid < residualLen)
	output[shared.task.residualOffs + pos + tid] = shared.data[tid + shared.task.residualOrder] - (sum >> shared.task.shift);
}
#endif
