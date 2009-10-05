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
    int abits;
    int porder;
    int reserved[2];
    int coefs[32]; // fixme: should be short?
} encodeResidualTaskStruct;

#define SUM16(buf,tid,op)   buf[tid] op buf[tid + 8]; buf[tid] op buf[tid + 4]; buf[tid] op buf[tid + 2]; buf[tid] op buf[tid + 1];
#define SUM32(buf,tid,op)   buf[tid] op buf[tid + 16]; SUM16(buf,tid,op)
#define SUM64(buf,tid,op)   if (tid < 32) buf[tid] op buf[tid + 32]; __syncthreads(); if (tid < 32) { SUM32(buf,tid,op) }
#define SUM128(buf,tid,op)  if (tid < 64) buf[tid] op buf[tid + 64]; __syncthreads(); SUM64(buf,tid,op)
#define SUM256(buf,tid,op)  if (tid < 128) buf[tid] op buf[tid + 128]; __syncthreads(); SUM128(buf,tid,op)
#define SUM512(buf,tid,op)  if (tid < 256) buf[tid] op buf[tid + 256]; __syncthreads(); SUM256(buf,tid,op)

#define FSQR(s) ((s)*(s))

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
	volatile int abits[256];
	encodeResidualTaskStruct task;
    } shared;

    if (threadIdx.x < 16)
	((int*)&shared.task)[threadIdx.x] = ((int*)(&tasks[blockIdx.x * tasksPerChannel]))[threadIdx.x];
    shared.wbits[threadIdx.x] = 0;
    shared.abits[threadIdx.x] = 0;
    __syncthreads();

    for (int pos = 0; pos < blocksize; pos += blockDim.x)
    {
	int smp = pos + threadIdx.x < blocksize ? samples[shared.task.samplesOffs + pos + threadIdx.x] : 0;
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
	tasks[blockIdx.x * tasksPerChannel + threadIdx.x].wbits = shared.task.wbits;
    if (threadIdx.x < tasksPerChannel)
	tasks[blockIdx.x * tasksPerChannel + threadIdx.x].abits = shared.task.abits;
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
	encodeResidualTaskStruct task2;
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
    __syncthreads();
    if (tid < sizeof(shared.task2) / sizeof(int))
	((int*)&shared.task2)[tid] = ((int*)(output + shared.task.residualOffs))[tid];
    __syncthreads();
    
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
	    //int precision = 13 - (shared.task.blocksize <= 2304) - (shared.task.blocksize <= 1152) - (shared.task.blocksize <= 576);
	    int precision = max(3, min(13 - (shared.task.blocksize <= 2304) - (shared.task.blocksize <= 1152) - (shared.task.blocksize <= 576), shared.task2.abits));
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
	    shared.bits[tid] = __mul24(33 - __clz(coef ^ (coef >> 31)), tid <= order);
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

extern "C" __global__ void cudaComputeLPCLattice(
    encodeResidualTaskStruct *tasks,
    const int taskCount, // tasks per block
    const int *samples,
    const int precisions,
    const int max_order // should be <= 12
)
{
    __shared__ struct {
	volatile encodeResidualTaskStruct task;
	volatile float F[512];
	volatile float lpc[12][32];
	union {
	    volatile float tmp[256];
	    volatile int tmpi[256];
	};
    } shared;

    // fetch task data
    if (threadIdx.x < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[threadIdx.x] = ((int*)(tasks + taskCount * blockIdx.y))[threadIdx.x];
    __syncthreads();

    // F = samples; B = samples
    //int frameSize = shared.task.blocksize;
    int s1 = threadIdx.x < shared.task.blocksize ? samples[shared.task.samplesOffs + threadIdx.x] : 0;
    int s2 = threadIdx.x + 256 < shared.task.blocksize ? samples[shared.task.samplesOffs + threadIdx.x + 256] : 0;
    shared.tmpi[threadIdx.x] = s1|s2;
    __syncthreads();
    SUM256(shared.tmpi,threadIdx.x,|=);
    if (threadIdx.x == 0)
	shared.task.wbits = max(0,__ffs(shared.tmpi[0]) - 1);
    __syncthreads();
    if (threadIdx.x < taskCount)
	tasks[blockIdx.y * taskCount + threadIdx.x].wbits = shared.task.wbits;
    shared.tmpi[threadIdx.x] = (s1 ^ (s1 >> 31)) | (s2 ^ (s2 >> 31));
    __syncthreads();
    SUM256(shared.tmpi,threadIdx.x,|=);
    if (threadIdx.x == 0)
	shared.task.abits = 32 - __clz(shared.tmpi[0]) - shared.task.wbits;
    __syncthreads();
    s1 >>= shared.task.wbits;
    s2 >>= shared.task.wbits;
    shared.F[threadIdx.x] = s1;
    shared.F[threadIdx.x + 256] = s2;
    __syncthreads();

    for (int order = 1; order <= max_order; order++)
    {
	// reff = F(order+1:frameSize) * B(1:frameSize-order)' / DEN
	float f1 = (threadIdx.x + order < shared.task.blocksize) * shared.F[threadIdx.x + order];
	float f2 = (threadIdx.x + 256 + order < shared.task.blocksize) * shared.F[threadIdx.x + 256 + order];
	s1 *= (threadIdx.x + order < shared.task.blocksize);
	s2 *= (threadIdx.x + 256 + order < shared.task.blocksize);

	// DEN = F(order+1:frameSize) * F(order+1:frameSize)' + B(1:frameSize-order) * B(1:frameSize-order)' (BURG)
	shared.tmp[threadIdx.x] = FSQR(f1) + FSQR(f2) + FSQR(s1) + FSQR(s2);
	__syncthreads();
	SUM256(shared.tmp, threadIdx.x, +=);
	__syncthreads();
	float DEN = shared.tmp[0] / 2;
	    //shared.PE[order-1] = shared.tmp[0] / 2 / (frameSize - order + 1);
	__syncthreads();

	shared.tmp[threadIdx.x] = f1 * s1 + f2 * s2;
	__syncthreads(); 
	SUM256(shared.tmp, threadIdx.x, +=);
	__syncthreads();
	float reff = shared.tmp[0] / DEN;
	__syncthreads();

	// arp(order) = rc(order) = reff
	if (threadIdx.x == 0)
	    shared.lpc[order - 1][order - 1] = reff;
	    //shared.rc[order - 1] = shared.lpc[order - 1][order - 1] = reff;

	// Levinson-Durbin recursion
	// arp(1:order-1) = arp(1:order-1) - reff * arp(order-1:-1:1)
	if (threadIdx.x < order - 1)
	    shared.lpc[order - 1][threadIdx.x] = shared.lpc[order - 2][threadIdx.x] - reff * shared.lpc[order - 2][order - 2 - threadIdx.x];

	// F1 = F(order+1:frameSize) - reff * B(1:frameSize-order)
	// B(1:frameSize-order) = B(1:frameSize-order) - reff * F(order+1:frameSize)
	// F(order+1:frameSize) = F1
	if (threadIdx.x < shared.task.blocksize - order)
	    shared.F[order + threadIdx.x] -= reff * s1;
	if (threadIdx.x + 256 < shared.task.blocksize - order)
	    shared.F[order + threadIdx.x + 256] -= reff * s2;
	s1 -= reff * f1;
	s2 -= reff * f2;
	__syncthreads();
    }
    // Quantization
    for (int order = (threadIdx.x >> 5); order < max_order; order += 8)
    for (int precision = 0; precision < precisions; precision++)
    {
	int cn = threadIdx.x & 31;
	// get 15 bits of each coeff
	int coef = cn <= order ? __float2int_rn(shared.lpc[order][cn] * (1 << 15)) : 0;
	// remove sign bits
	shared.tmpi[threadIdx.x] = coef ^ (coef >> 31);
	// OR reduction
	SUM32(shared.tmpi,threadIdx.x,|=);
	// choose precision	
	//int cbits = max(3, min(10, 5 + (shared.task.abits >> 1))); //  - __float2int_rn(shared.PE[order - 1])
	int cbits = max(3, min(10, shared.task.abits)) - precision;// + precision); //  - __float2int_rn(shared.PE[order - 1])
	// calculate shift based on precision and number of leading zeroes in coeffs
	int shift = max(0,min(15, __clz(shared.tmpi[threadIdx.x & ~31]) - 18 + cbits));
	//if (shared.task.abits + 32 - __clz(order) < shift
	//int shift = max(0,min(15, (shared.task.abits >> 2) - 14 + __clz(shared.tmpi[threadIdx.x & ~31]) + ((32 - __clz(order))>>1)));
	// quantize coeffs with given shift
	coef = cn <= order ? max(-(1 << (cbits - 1)), min((1 << (cbits - 1)) -1, __float2int_rn(shared.lpc[order][order - cn] * (1 << shift)))) : 0;
	// error correction
	//shared.tmp[threadIdx.x] = (threadIdx.x != 0) * (shared.arp[threadIdx.x - 1]*(1 << shared.task.shift) - shared.task.coefs[threadIdx.x - 1]);
	//shared.task.coefs[threadIdx.x] = max(-(1 << (shared.task.cbits - 1)), min((1 << (shared.task.cbits - 1))-1, __float2int_rn((shared.arp[threadIdx.x]) * (1 << shared.task.shift) + shared.tmp[threadIdx.x])));
	// remove sign bits
	shared.tmpi[threadIdx.x] = coef ^ (coef >> 31);
	// OR reduction
	SUM32(shared.tmpi,threadIdx.x,|=);
	// calculate actual number of bits (+1 for sign)
	cbits = 1 + 32 - __clz(shared.tmpi[threadIdx.x & ~31]);

	// output shift, cbits and output coeffs
	int taskNo = taskCount * blockIdx.y + order + precision * max_order;
	if (cn == 0)
	    tasks[taskNo].shift = shift;
	if (cn == 0)
	    tasks[taskNo].cbits = cbits;
	if (cn <= order)
	    tasks[taskNo].coefs[cn] = coef;
    }
}

//extern "C" __global__ void cudaComputeLPCLattice512(
//    encodeResidualTaskStruct *tasks,
//    const int taskCount, // tasks per block
//    const int *samples,
//    const int frameSize, // <= 512
//    const int max_order // should be <= 32
//)
//{
//    __shared__ struct {
//	encodeResidualTaskStruct task;
//	float F[512];
//	float B[512];
//	float lpc[32][32];
//	volatile float tmp[512];
//	volatile float arp[32];
//	volatile float rc[32];
//	volatile int   bits[512];
//	volatile float f, b;
//    } shared;
//
//    // fetch task data
//    if (threadIdx.x < sizeof(shared.task) / sizeof(int))
//	((int*)&shared.task)[threadIdx.x] = ((int*)(tasks + taskCount * blockIdx.y))[threadIdx.x];    
//    __syncthreads();
//
//    // F = samples; B = samples
//    shared.F[threadIdx.x] = threadIdx.x < frameSize ? samples[shared.task.samplesOffs + threadIdx.x] >> shared.task.wbits : 0.0f;
//    shared.B[threadIdx.x] = shared.F[threadIdx.x];
//    __syncthreads();
//
//    // DEN = F*F'
//    shared.tmp[threadIdx.x] = FSQR(shared.F[threadIdx.x]);
//    __syncthreads();
//    SUM512(shared.tmp,threadIdx.x,+=);
//    __syncthreads();
//    if (threadIdx.x == 0)
//	shared.f = shared.b = shared.tmp[0];
// //   if (threadIdx.x == 0)
//	//shared.PE[0] = DEN / frameSize;
//    __syncthreads();
//
//    for (int order = 1; order <= max_order; order++)
//    {
//	// reff = F(order+1:frameSize) * B(1:frameSize-order)' / DEN
//	shared.tmp[threadIdx.x] = (threadIdx.x + order < frameSize) * shared.F[threadIdx.x + order] * shared.B[threadIdx.x];
//	__syncthreads(); 
//	SUM512(shared.tmp, threadIdx.x,+=);
//	__syncthreads();
//	
//	//float reff = shared.tmp[0] * rsqrtf(shared.b * shared.f); // Geometric lattice
//	float reff = shared.tmp[0] * 2 / (shared.b + shared.f); // Burg method
//	__syncthreads();
//
//	// Levinson-Durbin recursion
//	// arp(order) = rc(order) = reff
//	// arp(1:order-1) = arp(1:order-1) - reff * arp(order-1:-1:1)
//	if (threadIdx.x == 32)
//	    shared.arp[order - 1] = shared.rc[order - 1] = reff;
//	if (threadIdx.x < 32)
//	    shared.arp[threadIdx.x] -= (threadIdx.x < order - 1) * __fmul_rz(reff, shared.arp[order - 2 - threadIdx.x]);
//
//	// F1 = F(order+1:frameSize) - reff * B(1:frameSize-order)
//	// B(1:frameSize-order) = B(1:frameSize-order) - reff * F(order+1:frameSize)
//	// F(order+1:frameSize) = F1
//	if (threadIdx.x < frameSize - order)
//	{
//	    float f;// = shared.F[threadIdx.x + order];
//	    shared.F[threadIdx.x + order] = (f = shared.F[threadIdx.x + order]) - reff * shared.B[threadIdx.x];
//	    shared.B[threadIdx.x] -= reff * f;
//	}
//	__syncthreads();
//
//	// f = F(order+1:frameSize) * F(order+1:frameSize)'
//	// b = B(1:frameSize-order) * B(1:frameSize-order)'
//	shared.tmp[threadIdx.x] = (threadIdx.x < frameSize - order) * FSQR(shared.F[threadIdx.x + order]);
//	__syncthreads();
//	SUM512(shared.tmp, threadIdx.x,+=);
//	__syncthreads();
//	if (threadIdx.x == 0)
//	    shared.f = shared.tmp[0];
//	__syncthreads();
//
//	shared.tmp[threadIdx.x] = (threadIdx.x < frameSize - order) * FSQR(shared.B[threadIdx.x]);
//	__syncthreads();
//	SUM512(shared.tmp, threadIdx.x,+=);
//	__syncthreads();
//	if (threadIdx.x == 0)
//	    shared.b = shared.tmp[0];
//	__syncthreads();
//
//	if (threadIdx.x < 32)
//	    shared.lpc[order - 1][threadIdx.x] = shared.arp[threadIdx.x];
//
//	//if (threadIdx.x == 0)
//	//    shared.PE[order] = (shared.b + shared.f) / 2 / (frameSize - order);
//	__syncthreads();
//    }
//    for (int order = 1 + (threadIdx.x >> 5); order <= max_order; order += 16)
//    {
//	// Quantization
//	int cn = threadIdx.x & 31;
//	int precision = 10 - (order > 8) - min(2, shared.task.wbits);
//	int taskNo = taskCount * blockIdx.y + order - 1;
//	shared.bits[threadIdx.x] = __mul24((33 - __clz(__float2int_rn(fabs(shared.lpc[order - 1][cn]) * (1 << 15))) - precision), cn < order);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 16]);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 8]);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 4]);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 2]);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 1]);
//	int sh = max(0,min(15, 15 - shared.bits[threadIdx.x - cn]));
//            
//	// reverse coefs
//	int coef = max(-(1 << precision),min((1 << precision)-1,__float2int_rn(shared.lpc[order - 1][order - 1 - cn] * (1 << sh))));
//	if (cn < order)
//	    tasks[taskNo].coefs[cn] = coef;
//	if (cn == 0)
//	    tasks[taskNo].shift = sh;
//	shared.bits[threadIdx.x] = __mul24(33 - max(__clz(coef),__clz(-1 ^ coef)), cn < order);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 16]);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 8]);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 4]);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 2]);
//	shared.bits[threadIdx.x] = max(shared.bits[threadIdx.x], shared.bits[threadIdx.x + 1]);
//	int cbits = shared.bits[threadIdx.x - cn];
//	if (cn == 0)
//	    tasks[taskNo].cbits = cbits;
//    }
//}

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

extern "C" __global__ void cudaChooseBestMethod(
    encodeResidualTaskStruct *tasks,
    int *residual,
    int partCount, // <= blockDim.y (256)
    int taskCount
    )
{
    __shared__ struct {
	volatile int index[128];
	volatile int length[256];
	volatile int partLen[256];
	volatile encodeResidualTaskStruct task[8];
    } shared;
    const int tid = threadIdx.x + threadIdx.y * 32;
    
    shared.length[tid] = 0x7fffffff;
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
			shared.task[threadIdx.y].type == Fixed ? shared.task[threadIdx.y].residualOrder * obits + 6 + (4 * partCount/2) + shared.partLen[threadIdx.y * 32] :
			shared.task[threadIdx.y].type == LPC ? shared.task[threadIdx.y].residualOrder * obits + 4 + 5 + shared.task[threadIdx.y].residualOrder * shared.task[threadIdx.y].cbits + 6 + (4 * partCount/2)/* << porder */ + shared.partLen[threadIdx.y * 32] :
			shared.task[threadIdx.y].type == Constant ? obits * (1 + shared.task[threadIdx.y].blocksize * (shared.partLen[threadIdx.y * 32] != 0)) : 
			obits * shared.task[threadIdx.y].blocksize);
	    }
	}
    //shared.index[threadIdx.x] = threadIdx.x;
    //shared.length[threadIdx.x] = (threadIdx.x < taskCount) ? tasks[threadIdx.x + taskCount * blockIdx.y].size : 0x7fffffff;

    __syncthreads();

    if (tid < taskCount)
	tasks[tid + taskCount * blockIdx.y].size = shared.length[tid];

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
	    tasks[taskCount * blockIdx.y].best_index = taskCount * blockIdx.y + shared.index[shared.length[1] < shared.length[0]];
    }
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
    __syncthreads();
    shared.data[tid + shared.task.residualOrder] -= (sum >> shared.task.shift);
    __syncthreads();
    if (tid >= shared.task.residualOrder && tid < residualLen + shared.task.residualOrder)
	output[shared.task.residualOffs + pos + tid] = shared.data[tid];
    if (tid + 256 < residualLen + shared.task.residualOrder)
	output[shared.task.residualOffs + pos + tid + 256] = shared.data[tid + 256];
}

extern "C" __global__ void cudaCalcPartition(
    int* partition_lengths,
    int* residual,
    int* samples,
    encodeResidualTaskStruct *tasks,
    int max_porder, // <= 8
    int psize, // == (shared.task.blocksize >> max_porder), < 256
    int parts_per_block // == 256 / psize, > 0, <= 16
    )
{
    __shared__ struct {
	int data[256+32];
	encodeResidualTaskStruct task;
    } shared;
    const int tid = threadIdx.x + (threadIdx.y << 4);
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(&tasks[blockIdx.y]))[tid];
    __syncthreads();

    const int parts = min(parts_per_block, (1 << max_porder) - blockIdx.x * parts_per_block);
    const int offs = blockIdx.x * psize * parts_per_block + tid;

    // fetch samples
    if (tid < 32) shared.data[tid] = min(offs, tid + shared.task.residualOrder) >= 32 ? samples[shared.task.samplesOffs + offs - 32] >> shared.task.wbits : 0;
    shared.data[32 + tid] = tid < parts * psize ? samples[shared.task.samplesOffs + offs] >> shared.task.wbits : 0;
    __syncthreads();

    // compute residual
    int s = 0;
    for (int c = -shared.task.residualOrder; c < 0; c++)
	s += __mul24(shared.data[32 + tid + c], shared.task.coefs[shared.task.residualOrder + c]);
    s = shared.data[32 + tid] - (s >> shared.task.shift);

    if (offs >= shared.task.residualOrder && tid < parts * psize)
	residual[shared.task.residualOffs + offs] = s;
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

    s = (psize - shared.task.residualOrder * (threadIdx.x + blockIdx.x == 0)) * (threadIdx.y + 1);
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
    encodeResidualTaskStruct *tasks,
    int max_porder, // <= 8
    int psize, // == 16
    int parts_per_block // == 16
    )
{
    __shared__ struct {
	int data[256+32];
	encodeResidualTaskStruct task;
    } shared;
    const int tid = threadIdx.x + (threadIdx.y << 4);
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(&tasks[blockIdx.y]))[tid];
    __syncthreads();

    const int offs = (blockIdx.x << 8) + tid;

    // fetch samples
    if (tid < 32) shared.data[tid] = min(offs, tid + shared.task.residualOrder) >= 32 ? samples[shared.task.samplesOffs + offs - 32] >> shared.task.wbits : 0;
    shared.data[32 + tid] = samples[shared.task.samplesOffs + offs] >> shared.task.wbits;
    __syncthreads();

    // compute residual
    int s = 0;
    for (int c = -shared.task.residualOrder; c < 0; c++)
	s += __mul24(shared.data[32 + tid + c], shared.task.coefs[shared.task.residualOrder + c]);
    s = shared.data[32 + tid] - (s >> shared.task.shift);

    if (offs >= shared.task.residualOrder)
	residual[shared.task.residualOffs + offs] = s;
    else
	s = 0;

    // convert to unsigned
    s = min(0xfffff, (s << 1) ^ (s >> 31));
    __syncthreads();
    shared.data[tid + threadIdx.y] = s;
    __syncthreads();

    // calc number of unary bits for part threadIdx.x with rice paramater threadIdx.y
    int dpos = __mul24(threadIdx.x, 17);
    s =
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
	partition_lengths[pos] = s + (16 - shared.task.residualOrder * (threadIdx.x + blockIdx.x == 0)) * (threadIdx.y + 1);
}

extern "C" __global__ void cudaCalcLargePartition(
    int* partition_lengths,
    int* residual,
    int* samples,
    encodeResidualTaskStruct *tasks,
    int max_porder, // <= 8
    int psize, // == >= 128
    int parts_per_block // == 1
    )
{
    __shared__ struct {
	int data[256];
	volatile int length[256];
	encodeResidualTaskStruct task;
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
	int s = (offs >= shared.task.residualOrder && pos + tid < psize) ? residual[shared.task.residualOffs + offs] : 0;
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
	partition_lengths[pos + blockIdx.x] = min(0xfffff,shared.length[tid]) + (psize - shared.task.residualOrder * (blockIdx.x == 0)) * (threadIdx.y + 1);
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
    encodeResidualTaskStruct *tasks,
    int* rice_parameters,
    int max_porder
    )
{
    __shared__ struct {
	int data[512];
	volatile int tmp[256];
	int length[32];
	int index[32];
	encodeResidualTaskStruct task;
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
	    tasks[blockIdx.y].porder = shared.index[0];
	if (threadIdx.x == 0)
	{
	    int obits = shared.task.obits - shared.task.wbits;	    
	    tasks[blockIdx.y].size =
		shared.task.type == Fixed ? shared.task.residualOrder * obits + 6 + l1 :
		shared.task.type == LPC ? shared.task.residualOrder * obits + 6 + l1 + 4 + 5 + shared.task.residualOrder * shared.task.cbits :
		shared.task.type == Constant ? obits : obits * shared.task.blocksize;
	}
    }
    __syncthreads();
    int porder = shared.index[0];
    //shared.data[threadIdx.x] = threadIdx.x < (1 << porder) ? rice_parameters[pos - (2 << porder) + threadIdx.x] : 0;
    if (threadIdx.x < (1 << porder))
	best_rice_parameters[(blockIdx.y << max_porder) + threadIdx.x] = rice_parameters[pos - (2 << porder) + threadIdx.x];
    // FIXME: should be bytes?
}

#endif
