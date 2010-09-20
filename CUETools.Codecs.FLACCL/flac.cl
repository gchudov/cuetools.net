/**
 * CUETools.FLACCL: FLAC audio encoder using OpenCL
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

#ifndef _FLACCL_KERNEL_H_
#define _FLACCL_KERNEL_H_

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
} FLACCLSubframeData;

typedef struct
{
    FLACCLSubframeData data;
    union 
    {
	int coefs[32]; // fixme: should be short?
	int4 coefs4[8];
    };
} FLACCLSubframeTask;

__kernel void cudaStereoDecorr(
    __global int *samples,
    __global short2 *src,
    int offset
)
{
    int pos = get_global_id(0);
    if (pos < offset)
    {
	short2 s = src[pos];
	samples[pos] = s.x;
	samples[1 * offset + pos] = s.y;
	samples[2 * offset + pos] = (s.x + s.y) >> 1;
	samples[3 * offset + pos] = s.x - s.y;
    }
}

__kernel void cudaChannelDecorr2(
    __global int *samples,
    __global short2 *src,
    int offset
)
{
    int pos = get_global_id(0);
    if (pos < offset)
    {
	short2 s = src[pos];
	samples[pos] = s.x;
	samples[1 * offset + pos] = s.y;
    }
}

//__kernel void cudaChannelDecorr(
//    int *samples,
//    short *src,
//    int offset
//)
//{
//    int pos = get_global_id(0);
//    if (pos < offset)
//	samples[get_group_id(1) * offset + pos] = src[pos * get_num_groups(1) + get_group_id(1)];
//}

#define __ffs(a) (32 - clz(a & (-a)))
//#define __ffs(a) (33 - clz(~a & (a - 1)))

__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void cudaFindWastedBits(
    __global FLACCLSubframeTask *tasks,
    __global int *samples,
    int tasksPerChannel
)
{
    __local volatile int wbits[128];
    __local volatile int abits[128];
    __local FLACCLSubframeData task;

    int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0) * tasksPerChannel].data))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int w = 0, a = 0;
    for (int pos = 0; pos < task.blocksize; pos += get_local_size(0))
    {
	int smp = pos + tid < task.blocksize ? samples[task.samplesOffs + pos + tid] : 0;
	w |= smp;
	a |= smp ^ (smp >> 31);
    }
    wbits[tid] = w;
    abits[tid] = a;
    barrier(CLK_LOCAL_MEM_FENCE);
    //atom_or(shared.wbits, shared.wbits[tid]);
    //atom_or(shared.abits, shared.abits[tid]);
    //SUM256(shared.wbits, tid, |=);
    //SUM256(shared.abits, tid, |=);
    //SUM128(wbits, tid, |=);
    //SUM128(abits, tid, |=);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1)
    {
	if (tid < s)
	{
	    wbits[tid] |= wbits[tid + s];
	    abits[tid] |= abits[tid + s];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (tid == 0)
	task.wbits = max(0,__ffs(wbits[0]) - 1);
    if (tid == 0)
	task.abits = 32 - clz(abits[0]) - task.wbits;
 //   if (tid == 0)
	//task.wbits = get_num_groups(0);
 //   if (tid == 0)
	//task.abits = get_local_size(0);
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < tasksPerChannel)
	tasks[get_group_id(0) * tasksPerChannel + tid].data.wbits = task.wbits;
    if (tid < tasksPerChannel)
	tasks[get_group_id(0) * tasksPerChannel + tid].data.abits = task.abits;
}

//__kernel __attribute__((reqd_work_group_size(32, 4, 1)))
//void cudaComputeAutocor(
//    __global float *output,
//    __global const int *samples,
//    __global const float *window,
//    __global FLACCLSubframeTask *tasks,
//    const int max_order, // should be <= 32
//    const int windowCount, // windows (log2: 0,1)
//    const int taskCount // tasks per block
//)
//{
//    __local struct {
//	float data[256];
//	volatile float product[128];
//	FLACCLSubframeData task;
//	volatile int dataPos;
//	volatile int dataLen;
//    } shared;
//    const int tid = get_local_id(0) + get_local_id(1) * 32;
//    // fetch task data
//    if (tid < sizeof(shared.task) / sizeof(int))
//	((__local int*)&shared.task)[tid] = ((__global int*)(tasks + taskCount * (get_group_id(1) >> windowCount)))[tid];
//    if (tid == 0) 
//    {
//	shared.dataPos = get_group_id(0) * 7 * 32;
//	shared.dataLen = min(shared.task.blocksize - shared.dataPos, 7 * 32 + max_order);
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    // fetch samples
//    shared.data[tid] = tid < shared.dataLen ? samples[tid] * window[tid]: 0.0f;
//    int tid2 = tid + 128;
//    shared.data[tid2] = tid2 < shared.dataLen ? samples[tid2] * window[tid2]: 0.0f;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for (int lag = 0; lag <= max_order; lag ++)
//    {
//	if (lag <= 12)
//	    shared.product[tid] = 0.0f;
//	barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//    if (tid <= max_order)
//	output[(get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (max_order + 1) + tid] = shared.product[tid];
//}

__kernel __attribute__((reqd_work_group_size(32, 4, 1)))
void cudaComputeAutocor(
    __global float *output,
    __global const int *samples,
    __global const float *window,
    __global FLACCLSubframeTask *tasks,
    const int max_order, // should be <= 32
    const int windowCount, // windows (log2: 0,1)
    const int taskCount // tasks per block
)
{
    __local struct {
	float data[256];
	volatile float product[128];
	FLACCLSubframeData task;
	volatile float result[33];
	volatile int dataPos;
	volatile int dataLen;
	volatile int windowOffs;
	volatile int samplesOffs;
	//volatile int resultOffs;
    } shared;
    const int tid = get_local_id(0) + get_local_id(1) * 32;
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((__local int*)&shared.task)[tid] = ((__global int*)(tasks + taskCount * (get_group_id(1) >> windowCount)))[tid];
    if (tid == 0) 
    {
	shared.dataPos = get_group_id(0) * 7 * 32;
	shared.windowOffs = (get_group_id(1) & ((1 << windowCount)-1)) * shared.task.blocksize + shared.dataPos;
	shared.samplesOffs = shared.task.samplesOffs + shared.dataPos;
	shared.dataLen = min(shared.task.blocksize - shared.dataPos, 7 * 32 + max_order);
    }
    //if (tid == 32)
	//shared.resultOffs = __mul24(get_group_id(0) + __mul24(get_group_id(1), get_num_groups(0)), max_order + 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    // fetch samples
    shared.data[tid] = tid < shared.dataLen ? samples[shared.samplesOffs + tid] * window[shared.windowOffs + tid]: 0.0f;
    int tid2 = tid + 128;
    shared.data[tid2] = tid2 < shared.dataLen ? samples[shared.samplesOffs + tid2] * window[shared.windowOffs + tid2]: 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);

    const int ptr = get_local_id(0) * 7;
    //if (get_local_id(1) == 0) for (int lag = 0; lag <= max_order; lag ++)
    //for (int lag = get_local_id(1); lag <= max_order; lag += get_local_size(1))
    for (int lag0 = 0; lag0 <= max_order; lag0 += get_local_size(1))
    {
	////const int productLen = min(shared.task.blocksize - get_group_id(0) * partSize - lag, partSize);
	const int lag = lag0 + get_local_id(1);
	const int ptr2 = ptr + lag;
	shared.product[tid] =
	    shared.data[ptr + 0] * shared.data[ptr2 + 0] +
	    shared.data[ptr + 1] * shared.data[ptr2 + 1] +
	    shared.data[ptr + 2] * shared.data[ptr2 + 2] +
	    shared.data[ptr + 3] * shared.data[ptr2 + 3] +
	    shared.data[ptr + 4] * shared.data[ptr2 + 4] +
	    shared.data[ptr + 5] * shared.data[ptr2 + 5] +
	    shared.data[ptr + 6] * shared.data[ptr2 + 6];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int l = 16; l > 1; l >>= 1)
	{
	    if (get_local_id(0) < l)
		shared.product[tid] = shared.product[tid] + shared.product[tid + l];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	// return results
	if (get_local_id(0) == 0 && lag <= max_order)
	    shared.result[lag] = shared.product[tid] + shared.product[tid + 1];
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid <= max_order)
	output[(get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (max_order + 1) + tid] = shared.result[tid];
    //output[(get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (max_order + 1) + tid] = shared.product[tid];
    //output[(get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (max_order + 1) + tid] = shared.windowOffs;
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void cudaComputeLPC(
    __global FLACCLSubframeTask *tasks,
    int taskCount, // tasks per block
    __global float*autoc,
    int max_order, // should be <= 32
    __global float *lpcs,
    int windowCount,
    int partCount
)
{
    __local struct {
	FLACCLSubframeData task;
	volatile float parts[32];
	volatile float ldr[32];
	volatile float gen1[32];
	volatile float error[32];
	volatile float autoc[33];
	volatile int lpcOffs;
	volatile int autocOffs;
    } shared;
    const int tid = get_local_id(0);// + get_local_id(1) * 32;
    
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((__local int*)&shared.task)[tid] = ((__global int*)(tasks + get_group_id(1) * taskCount))[tid];
    if (tid == 0)
    {
	shared.lpcOffs = (get_group_id(0) + get_group_id(1) * windowCount) * (max_order + 1) * 32;
	shared.autocOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (max_order + 1) * partCount;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // add up autocorrelation parts

 //   for (int order = get_local_id(0); order <= max_order; order += 32)
 //   {
	//float sum = 0.0f;
	//for (int pos = 0; pos < partCount; pos++)
	//    sum += autoc[shared.autocOffs + pos * (max_order + 1) + order];
	//shared.autoc[order] = sum;
 //   }

    for (int order = 0; order <= max_order; order ++)
    {
	float part = 0.0f;
	for (int pos = get_local_id(0); pos < partCount; pos += get_local_size(0))
	    part += autoc[shared.autocOffs + pos * (max_order + 1) + order];
	shared.parts[tid] = part;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int l = get_local_size(0) / 2; l > 1; l >>= 1)
	{
	    if (get_local_id(0) < l)
		shared.parts[tid] += shared.parts[tid + l];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (get_local_id(0) == 0)
	    shared.autoc[order] = shared.parts[tid] + shared.parts[tid + 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute LPC using Schur and Levinson-Durbin recursion
    float gen0 = shared.gen1[get_local_id(0)] = shared.autoc[get_local_id(0)+1];
    shared.ldr[get_local_id(0)] = 0.0f;
    float error = shared.autoc[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int order = 0; order < max_order; order++)
    {
	// Schur recursion
	float reff = -shared.gen1[0] / error;
	error += shared.gen1[0] * reff; // Equivalent to error *= (1 - reff * reff);
	float gen1;
	if (get_local_id(0) < max_order - 1 - order)
	{
	    gen1 = shared.gen1[get_local_id(0) + 1] + reff * gen0;
	    gen0 += shared.gen1[get_local_id(0) + 1] * reff;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) < max_order - 1 - order)
	    shared.gen1[get_local_id(0)] = gen1;

	// Store prediction error
	if (get_local_id(0) == 0)
	    shared.error[order] = error;

	// Levinson-Durbin recursion
	float ldr = 
	    select(0.0f, reff * shared.ldr[order - 1 - get_local_id(0)], get_local_id(0) < order) +
	    select(0.0f, reff, get_local_id(0) == order);
        barrier(CLK_LOCAL_MEM_FENCE);
	shared.ldr[get_local_id(0)] += ldr;
        barrier(CLK_LOCAL_MEM_FENCE);

	// Output coeffs
	if (get_local_id(0) <= order)
	    lpcs[shared.lpcOffs + order * 32 + get_local_id(0)] = -shared.ldr[order - get_local_id(0)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Output prediction error estimates
    if (get_local_id(0) < max_order)
	lpcs[shared.lpcOffs + max_order * 32 + get_local_id(0)] = shared.error[get_local_id(0)];
}

//__kernel void cudaComputeLPCLattice(
//    FLACCLSubframeTask *tasks,
//    const int taskCount, // tasks per block
//    const int *samples,
//    const int windowCount,
//    const int max_order, // should be <= 12
//    float*lpcs
//)
//{
//    __local struct {
//	volatile FLACCLSubframeData task;
//	volatile float F[512];
//	volatile float arp[32];
//	volatile float tmp[256];
//	volatile float error[32];
//	volatile int lpcOffs;
//    } shared;
//
//    // fetch task data
//    if (get_local_id(0) < sizeof(shared.task) / sizeof(int))
//	((int*)&shared.task)[get_local_id(0)] = ((int*)(tasks + taskCount * get_group_id(1)))[get_local_id(0)];
//    if (get_local_id(0) == 0)
//	shared.lpcOffs = __mul24(__mul24(get_group_id(1) + 1, windowCount) - 1, max_order + 1) * 32;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    // F = samples; B = samples
//    float s1 = get_local_id(0) < shared.task.blocksize ? (samples[shared.task.samplesOffs + get_local_id(0)]) / 32768.0f : 0.0f;
//    float s2 = get_local_id(0) + 256 < shared.task.blocksize ? (samples[shared.task.samplesOffs + get_local_id(0) + 256]) / 32768.0f : 0.0f;
//    shared.F[get_local_id(0)] = s1;
//    shared.F[get_local_id(0) + 256] = s2;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    shared.tmp[get_local_id(0)] = FSQR(s1) + FSQR(s2);
//    barrier(CLK_LOCAL_MEM_FENCE);
//    SUM256(shared.tmp, get_local_id(0), +=);
//    barrier(CLK_LOCAL_MEM_FENCE);
//    float DEN = shared.tmp[0];
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for (int order = 0; order < max_order; order++)
//    {
//	// reff = F(order+1:frameSize) * B(1:frameSize-order)' / DEN
//	int idxF = get_local_id(0) + order + 1;
//	int idxF2 = idxF + 256;
//
//	shared.tmp[get_local_id(0)] = idxF < shared.task.blocksize ? shared.F[idxF] * s1 : 0.0f;
//	shared.tmp[get_local_id(0)] += idxF2 < shared.task.blocksize ? shared.F[idxF2] * s2 : 0.0f;
//	barrier(CLK_LOCAL_MEM_FENCE); 
//	SUM256(shared.tmp, get_local_id(0), +=);
//	barrier(CLK_LOCAL_MEM_FENCE);
//	float reff = shared.tmp[0] / DEN;
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	// arp(order) = rc(order) = reff
//	if (get_local_id(0) == 0)
//	    shared.arp[order] = reff;
//	    //shared.rc[order - 1] = shared.lpc[order - 1][order - 1] = reff;
//
//	// Levinson-Durbin recursion
//	// arp(1:order-1) = arp(1:order-1) - reff * arp(order-1:-1:1)
//	if (get_local_id(0) < order)
//	    shared.arp[get_local_id(0)] = shared.arp[get_local_id(0)] - reff * shared.arp[order - 1 - get_local_id(0)];
//	
//	// Output coeffs
//	if (get_local_id(0) <= order)
//	    lpcs[shared.lpcOffs + order * 32 + get_local_id(0)] = shared.arp[order - get_local_id(0)];
//
//	// F1 = F(order+1:frameSize) - reff * B(1:frameSize-order)
//	// B(1:frameSize-order) = B(1:frameSize-order) - reff * F(order+1:frameSize)
//	// F(order+1:frameSize) = F1
//	if (idxF < shared.task.blocksize)
//	{
//	    float f1 = shared.F[idxF];
//	    shared.F[idxF] -= reff * s1;
//	    s1 -= reff * f1;
//	}
//	if (idxF2 < shared.task.blocksize)
//	{
//	    float f2 = shared.F[idxF2];
//	    shared.F[idxF2] -= reff * s2;
//	    s2 -= reff * f2;
//	}
//
//	// DEN = F(order+1:frameSize) * F(order+1:frameSize)' + B(1:frameSize-order) * B(1:frameSize-order)' (BURG)
//	shared.tmp[get_local_id(0)] = (idxF + 1 < shared.task.blocksize ? FSQR(shared.F[idxF]) + FSQR(s1) : 0);
//	shared.tmp[get_local_id(0)] += (idxF2 + 1 < shared.task.blocksize ? FSQR(shared.F[idxF2]) + FSQR(s2) : 0);
//	barrier(CLK_LOCAL_MEM_FENCE);
//	SUM256(shared.tmp, get_local_id(0), +=);
//	barrier(CLK_LOCAL_MEM_FENCE);
//	DEN = shared.tmp[0] / 2;
//	// shared.PE[order-1] = shared.tmp[0] / 2 / (frameSize - order + 1);
//	if (get_local_id(0) == 0)
//	    shared.error[order] = DEN / (shared.task.blocksize - order);
//	barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
//    // Output prediction error estimates
//    if (get_local_id(0) < max_order)
//	lpcs[shared.lpcOffs + max_order * 32 + get_local_id(0)] = shared.error[get_local_id(0)];
//}

__kernel __attribute__((reqd_work_group_size(32, 4, 1)))
void cudaQuantizeLPC(
    __global FLACCLSubframeTask *tasks,
    int taskCount, // tasks per block
    int taskCountLPC, // tasks per set of coeffs (<= 32)
    __global float*lpcs,
    int max_order, // should be <= 32
    int minprecision,
    int precisions
)
{
    __local struct {
	FLACCLSubframeData task;
	volatile int tmpi[128];
	volatile int index[64];
	volatile float error[64];
	volatile int lpcOffs;
    } shared;

    const int tid = get_local_id(0) + get_local_id(1) * 32;

    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((__local int*)&shared.task)[tid] = ((__global int*)(tasks + get_group_id(1) * taskCount))[tid];
    if (tid == 0)
	shared.lpcOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (max_order + 1) * 32;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(1) == 0)
    {
	shared.index[get_local_id(0)] = min(max_order - 1, get_local_id(0));
	shared.error[get_local_id(0)] = shared.task.blocksize * 64 + get_local_id(0);
	shared.index[32 + get_local_id(0)] = min(max_order - 1, get_local_id(0));
	shared.error[32 + get_local_id(0)] = shared.task.blocksize * 64 + get_local_id(0);

        // Select best orders based on Akaike's Criteria

	// Load prediction error estimates
	if (get_local_id(0) < max_order)
	    shared.error[get_local_id(0)] = shared.task.blocksize * log(lpcs[shared.lpcOffs + max_order * 32 + get_local_id(0)]) + get_local_id(0) * 5.12f * log(shared.task.blocksize);
	    //shared.error[get_local_id(0)] = shared.task.blocksize * log(lpcs[shared.lpcOffs + max_order * 32 + get_local_id(0)]) + get_local_id(0) * 0.30f * (shared.task.abits + 1) * log(shared.task.blocksize);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Sort using bitonic sort
    for(int size = 2; size < 64; size <<= 1){
	//Bitonic merge
	int ddd = (get_local_id(0) & (size / 2)) == 0;
	for(int stride = size / 2; stride > 0; stride >>= 1){
	    int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
	    float e0, e1;
	    int i0, i1;
	    if (get_local_id(1) == 0)
	    {
		e0 = shared.error[pos];
		e1 = shared.error[pos + stride];
		i0 = shared.index[pos];
		i1 = shared.index[pos + stride];
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	    if ((e0 >= e1) == ddd && get_local_id(1) == 0)
	    {
		shared.error[pos] = e1;
		shared.error[pos + stride] = e0;
		shared.index[pos] = i1;
		shared.index[pos + stride] = i0;
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
    }

    //ddd == dir for the last bitonic merge step
    {
	for(int stride = 32; stride > 0; stride >>= 1){
	    //barrier(CLK_LOCAL_MEM_FENCE);
	    int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
	    float e0, e1;
	    int i0, i1;
	    if (get_local_id(1) == 0)
	    {
		e0 = shared.error[pos];
		e1 = shared.error[pos + stride];
		i0 = shared.index[pos];
		i1 = shared.index[pos + stride];
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	    if (e0 >= e1 && get_local_id(1) == 0)
	    {
		shared.error[pos] = e1;
		shared.error[pos + stride] = e0;
		shared.index[pos] = i1;
		shared.index[pos + stride] = i0;
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
    }

    // Quantization
    for (int ii = 0; ii < taskCountLPC; ii += get_local_size(1))
    {
	int i = ii + get_local_id(1);
	int order = shared.index[i >> precisions];
	float lpc = get_local_id(0) <= order ? lpcs[shared.lpcOffs + order * 32 + get_local_id(0)] : 0.0f;
	// get 15 bits of each coeff
	int coef = convert_int_rte(lpc * (1 << 15));
	// remove sign bits
	shared.tmpi[tid] = coef ^ (coef >> 31);
	barrier(CLK_LOCAL_MEM_FENCE);
	// OR reduction
	for (int l = get_local_size(0) / 2; l > 1; l >>= 1)
	{
	    if (get_local_id(0) < l)
		shared.tmpi[tid] |= shared.tmpi[tid + l];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	//SUM32(shared.tmpi,tid,|=);
	// choose precision	
	//int cbits = max(3, min(10, 5 + (shared.task.abits >> 1))); //  - convert_int_rte(shared.PE[order - 1])
	int cbits = max(3, min(min(13 - minprecision + (i - ((i >> precisions) << precisions)) - (shared.task.blocksize <= 2304) - (shared.task.blocksize <= 1152) - (shared.task.blocksize <= 576), shared.task.abits), clz(order) + 1 - shared.task.abits));
	// calculate shift based on precision and number of leading zeroes in coeffs
	int shift = max(0,min(15, clz(shared.tmpi[get_local_id(1) * 32] | shared.tmpi[get_local_id(1) * 32 + 1]) - 18 + cbits));

	//cbits = 13;
	//shift = 15;

	//if (shared.task.abits + 32 - clz(order) < shift
	//int shift = max(0,min(15, (shared.task.abits >> 2) - 14 + clz(shared.tmpi[get_local_id(0) & ~31]) + ((32 - clz(order))>>1)));
	// quantize coeffs with given shift
	coef = convert_int_rte(clamp(lpc * (1 << shift), -1 << (cbits - 1), 1 << (cbits - 1)));
	// error correction
	//shared.tmp[get_local_id(0)] = (get_local_id(0) != 0) * (shared.arp[get_local_id(0) - 1]*(1 << shared.task.shift) - shared.task.coefs[get_local_id(0) - 1]);
	//shared.task.coefs[get_local_id(0)] = max(-(1 << (shared.task.cbits - 1)), min((1 << (shared.task.cbits - 1))-1, convert_int_rte((shared.arp[get_local_id(0)]) * (1 << shared.task.shift) + shared.tmp[get_local_id(0)])));
	// remove sign bits
	shared.tmpi[tid] = coef ^ (coef >> 31);
	barrier(CLK_LOCAL_MEM_FENCE);
	// OR reduction
	for (int l = get_local_size(0) / 2; l > 1; l >>= 1)
	{
	    if (get_local_id(0) < l)
		shared.tmpi[tid] |= shared.tmpi[tid + l];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	//SUM32(shared.tmpi,tid,|=);
	// calculate actual number of bits (+1 for sign)
	cbits = 1 + 32 - clz(shared.tmpi[get_local_id(1) * 32] | shared.tmpi[get_local_id(1) * 32 + 1]);

	// output shift, cbits and output coeffs
	if (i < taskCountLPC)
	{
	    int taskNo = get_group_id(1) * taskCount + get_group_id(0) * taskCountLPC + i;
	    if (get_local_id(0) == 0)
		tasks[taskNo].data.shift = shift;
	    if (get_local_id(0) == 0)
		tasks[taskNo].data.cbits = cbits;
	    if (get_local_id(0) == 0)
		tasks[taskNo].data.residualOrder = order + 1;
	    if (get_local_id(0) <= order)
		tasks[taskNo].coefs[get_local_id(0)] = coef;
	}
    }
}

__kernel __attribute__(( vec_type_hint (int4)))
void cudaEstimateResidual(
    __global int*output,
    __global int*samples,
    __global FLACCLSubframeTask *tasks
    )
{
    __local float data[128 * 2];
    __local int residual[128];
    __local FLACCLSubframeTask task;
    __local float4 coefsf4[8];

    const int tid = get_local_id(0);
    if (tid < sizeof(task)/sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(1)]))[tid];
    barrier(CLK_GLOBAL_MEM_FENCE);

    int ro = task.data.residualOrder;
    int bs = task.data.blocksize;
    float res = 0;

    if (tid < 32)
	((__local float *)&coefsf4[0])[tid] = select(0.0f, ((float)task.coefs[tid]) / (1 << task.data.shift), tid < ro);
    data[tid] = tid < bs ? (float)(samples[task.data.samplesOffs + tid] >> task.data.wbits) : 0.0f;
    for (int pos = 0; pos < bs; pos += get_local_size(0))
    {
	// fetch samples
	float nextData = pos + tid + get_local_size(0) < bs ? (float)(samples[task.data.samplesOffs + pos + tid + get_local_size(0)] >> task.data.wbits) : 0.0f;
	data[tid + get_local_size(0)] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	// compute residual
	__local float4 * dptr = (__local float4 *)&data[tid];
	float sumf = data[tid + ro] - 
	    ( dot(dptr[0], coefsf4[0])
	    + dot(dptr[1], coefsf4[1])
#if MAX_ORDER > 8
	    + dot(dptr[2], coefsf4[2])
#if MAX_ORDER > 12
	    + dot(dptr[3], coefsf4[3])
#if MAX_ORDER > 16
	    + dot(dptr[4], coefsf4[4])
	    + dot(dptr[5], coefsf4[5])
	    + dot(dptr[6], coefsf4[6])
	    + dot(dptr[7], coefsf4[7])
#endif
#endif
#endif
	);
	//residual[tid] = sum;
	
	res += select(0.0f, min(fabs(sumf), (float)0x7fffff), pos + tid + ro < bs);
	barrier(CLK_LOCAL_MEM_FENCE);

	//int k = min(33 - clz(sum), 14);
	//res += select(0, 1 + k, pos + tid + ro < bs);
	
	//sum = residual[tid] + residual[tid + 1] + residual[tid + 2] + residual[tid + 3]
	//    + residual[tid + 4] + residual[tid + 5] + residual[tid + 6] + residual[tid + 7];
	//int k = clamp(29 - clz(sum), 0, 14);
	//res += select(0, 8 * (k + 1) + (sum >> k), pos + tid + ro < bs && !(tid & 7));
	
	data[tid] = nextData;
    }

    int residualLen = (bs - ro) / get_local_size(0) + select(0, 1, tid < (bs - ro) % get_local_size(0));
    int k = clamp(convert_int_rtn(log2((res + 0.000001f) / (residualLen + 0.000001f))), 0, 14);
    residual[tid] = residualLen * (k + 1) + (convert_int_rtz(res) >> k);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = get_local_size(0) / 2; l > 0; l >>= 1)
    {
	if (tid < l)
	    residual[tid] += residual[tid + l];
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
	output[get_group_id(1)] = residual[0];
}

__kernel void cudaChooseBestMethod(
    __global FLACCLSubframeTask *tasks,
    __global int *residual,
    int taskCount
    )
{
    __local struct {
	volatile int index[128];
	volatile int length[256];
	volatile FLACCLSubframeTask task[8];
    } shared;
    const int tid = get_local_id(0) + get_local_id(1) * 32;
    
    shared.length[tid] = 0x7fffffff;
    shared.index[tid] = tid;
    for (int task = 0; task < taskCount; task += get_local_size(1))
	if (task + get_local_id(1) < taskCount)
	{
	    // fetch task data
	    ((__local int*)&shared.task[get_local_id(1)])[get_local_id(0)] = 
		((__global int*)(tasks + task + get_local_id(1) + taskCount * get_group_id(1)))[get_local_id(0)];

	    barrier(CLK_LOCAL_MEM_FENCE);

	    if (get_local_id(0) == 0)
	    {
		// fetch part sum
		int partLen = residual[task + get_local_id(1) + taskCount * get_group_id(1)];
		//// calculate part size
		//int residualLen = shared.task[get_local_id(1)].data.blocksize - shared.task[get_local_id(1)].data.residualOrder;
		//residualLen = residualLen * (shared.task[get_local_id(1)].data.type != Constant || psum != 0);
		//// calculate rice parameter
		//int k = max(0, min(14, convert_int_rtz(log2((psum + 0.000001f) / (residualLen + 0.000001f) + 0.5f))));
		//// calculate part bit length
		//int partLen = residualLen * (k + 1) + (psum >> k);

		int obits = shared.task[get_local_id(1)].data.obits - shared.task[get_local_id(1)].data.wbits;
		shared.length[task + get_local_id(1)] =
		    min(obits * shared.task[get_local_id(1)].data.blocksize,
			shared.task[get_local_id(1)].data.type == Fixed ? shared.task[get_local_id(1)].data.residualOrder * obits + 6 + (4 * 1/2) + partLen :
			shared.task[get_local_id(1)].data.type == LPC ? shared.task[get_local_id(1)].data.residualOrder * obits + 4 + 5 + shared.task[get_local_id(1)].data.residualOrder * shared.task[get_local_id(1)].data.cbits + 6 + (4 * 1/2)/* << porder */ + partLen :
			shared.task[get_local_id(1)].data.type == Constant ? obits * (1 + shared.task[get_local_id(1)].data.blocksize * (partLen != 0)) : 
			obits * shared.task[get_local_id(1)].data.blocksize);
	    }
	}
    //shared.index[get_local_id(0)] = get_local_id(0);
    //shared.length[get_local_id(0)] = (get_local_id(0) < taskCount) ? tasks[get_local_id(0) + taskCount * get_group_id(1)].size : 0x7fffffff;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < taskCount)
	tasks[tid + taskCount * get_group_id(1)].data.size = shared.length[tid];

    int l1 = shared.length[tid];
    for (int sh = 8; sh > 0; sh --)
    {
	if (tid + (1 << sh) < get_local_size(0) * get_local_size(1))
	{
	    int l2 = shared.length[tid + (1 << sh)];
	    shared.index[tid] = shared.index[tid + ((l2 < l1) << sh)];
	    shared.length[tid] = l1 = min(l1, l2);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
	tasks[taskCount * get_group_id(1)].data.best_index = taskCount * get_group_id(1) + shared.index[shared.length[1] < shared.length[0]];
}

__kernel void cudaCopyBestMethod(
    __global FLACCLSubframeTask *tasks_out,
    __global FLACCLSubframeTask *tasks,
    int count
    )
{
    __local int best_index;
    if (get_local_id(0) == 0)
	best_index = tasks[count * get_group_id(1)].data.best_index;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < sizeof(FLACCLSubframeTask)/sizeof(int))
	((__global int*)(tasks_out + get_group_id(1)))[get_local_id(0)] = ((__global int*)(tasks + best_index))[get_local_id(0)];
}

__kernel void cudaCopyBestMethodStereo(
    __global FLACCLSubframeTask *tasks_out,
    __global FLACCLSubframeTask *tasks,
    int count
    )
{
    __local struct {
	int best_index[4];
	int best_size[4];
	int lr_index[2];
    } shared;
    if (get_local_id(0) < 4)
	shared.best_index[get_local_id(0)] = tasks[count * (get_group_id(1) * 4 + get_local_id(0))].data.best_index;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < 4)
	shared.best_size[get_local_id(0)] = tasks[shared.best_index[get_local_id(0)]].data.size;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
    {
	int bitsBest = shared.best_size[2] + shared.best_size[3]; // MidSide
	shared.lr_index[0] = shared.best_index[2];
	shared.lr_index[1] = shared.best_index[3];
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
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < sizeof(FLACCLSubframeTask)/sizeof(int))
	((__global int*)(tasks_out + 2 * get_group_id(1)))[get_local_id(0)] = ((__global int*)(tasks + shared.lr_index[0]))[get_local_id(0)];
    if (get_local_id(0) == 0)
	tasks_out[2 * get_group_id(1)].data.residualOffs = tasks[shared.best_index[0]].data.residualOffs;
    if (get_local_id(0) < sizeof(FLACCLSubframeTask)/sizeof(int))
	((__global int*)(tasks_out + 2 * get_group_id(1) + 1))[get_local_id(0)] = ((__global int*)(tasks + shared.lr_index[1]))[get_local_id(0)];
    if (get_local_id(0) == 0)
	tasks_out[2 * get_group_id(1) + 1].data.residualOffs = tasks[shared.best_index[1]].data.residualOffs;
}

//__kernel void cudaEncodeResidual(
//    int*output,
//    int*samples,
//    FLACCLSubframeTask *tasks
//    )
//{
//    __local struct {
//	int data[256 + 32];
//	FLACCLSubframeTask task;
//    } shared;
//    const int tid = get_local_id(0);
//    if (get_local_id(0) < sizeof(shared.task) / sizeof(int))
//	((int*)&shared.task)[get_local_id(0)] = ((int*)(&tasks[get_group_id(1)]))[get_local_id(0)];
//    barrier(CLK_LOCAL_MEM_FENCE);
//    const int partSize = get_local_size(0);
//    const int pos = get_group_id(0) * partSize;
//    const int dataLen = min(shared.task.data.blocksize - pos, partSize + shared.task.data.residualOrder);
//
//    // fetch samples
//    shared.data[tid] = tid < dataLen ? samples[shared.task.data.samplesOffs + pos + tid] >> shared.task.data.wbits : 0;
//    if (tid < 32) shared.data[tid + partSize] = tid + partSize < dataLen ? samples[shared.task.data.samplesOffs + pos + tid + partSize] >> shared.task.data.wbits : 0;
//    const int residualLen = max(0,min(shared.task.data.blocksize - pos - shared.task.data.residualOrder, partSize));
//
//    barrier(CLK_LOCAL_MEM_FENCE);    
//    // compute residual
//    int sum = 0;
//    for (int c = 0; c < shared.task.data.residualOrder; c++)
//	sum += __mul24(shared.data[tid + c], shared.task.coefs[c]);
//    barrier(CLK_LOCAL_MEM_FENCE);
//    shared.data[tid + shared.task.data.residualOrder] -= (sum >> shared.task.data.shift);
//    barrier(CLK_LOCAL_MEM_FENCE);
//    if (tid >= shared.task.data.residualOrder && tid < residualLen + shared.task.data.residualOrder)
//	output[shared.task.data.residualOffs + pos + tid] = shared.data[tid];
//    if (tid + 256 < residualLen + shared.task.data.residualOrder)
//	output[shared.task.data.residualOffs + pos + tid + 256] = shared.data[tid + 256];
//}
//
//__kernel void cudaCalcPartition(
//    int* partition_lengths,
//    int* residual,
//    int* samples,
//    FLACCLSubframeTask *tasks,
//    int max_porder, // <= 8
//    int psize, // == (shared.task.data.blocksize >> max_porder), < 256
//    int parts_per_block // == 256 / psize, > 0, <= 16
//    )
//{
//    __local struct {
//	int data[256+32];
//	FLACCLSubframeTask task;
//    } shared;
//    const int tid = get_local_id(0) + (get_local_id(1) << 4);
//    if (tid < sizeof(shared.task) / sizeof(int))
//	((int*)&shared.task)[tid] = ((int*)(&tasks[get_group_id(1)]))[tid];
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    const int parts = min(parts_per_block, (1 << max_porder) - get_group_id(0) * parts_per_block);
//    const int offs = get_group_id(0) * psize * parts_per_block + tid;
//
//    // fetch samples
//    if (tid < 32) shared.data[tid] = min(offs, tid + shared.task.data.residualOrder) >= 32 ? samples[shared.task.data.samplesOffs + offs - 32] >> shared.task.data.wbits : 0;
//    shared.data[32 + tid] = tid < parts * psize ? samples[shared.task.data.samplesOffs + offs] >> shared.task.data.wbits : 0;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    // compute residual
//    int s = 0;
//    for (int c = -shared.task.data.residualOrder; c < 0; c++)
//	s += __mul24(shared.data[32 + tid + c], shared.task.coefs[shared.task.data.residualOrder + c]);
//    s = shared.data[32 + tid] - (s >> shared.task.data.shift);
//
//    if (offs >= shared.task.data.residualOrder && tid < parts * psize)
//	residual[shared.task.data.residualOffs + offs] = s;
//    else
//	s = 0;
//
//    // convert to unsigned
//    s = min(0xfffff, (s << 1) ^ (s >> 31));
//
//    //barrier(CLK_LOCAL_MEM_FENCE);
//    //shared.data[tid] = s;
//    //barrier(CLK_LOCAL_MEM_FENCE);
//
//    //shared.data[tid] = (shared.data[tid] & (0x0000ffff << (tid & 16))) | (((shared.data[tid ^ 16] & (0x0000ffff << (tid & 16))) << (~tid & 16)) >> (tid & 16));
//    //shared.data[tid] = (shared.data[tid] & (0x00ff00ff << (tid & 8))) | (((shared.data[tid ^ 8] & (0x00ff00ff << (tid & 8))) << (~tid & 8)) >> (tid & 8));
//    //shared.data[tid] = (shared.data[tid] & (0x0f0f0f0f << (tid & 4))) | (((shared.data[tid ^ 4] & (0x0f0f0f0f << (tid & 4))) << (~tid & 4)) >> (tid & 4));
//    //shared.data[tid] = (shared.data[tid] & (0x33333333 << (tid & 2))) | (((shared.data[tid ^ 2] & (0x33333333 << (tid & 2))) << (~tid & 2)) >> (tid & 2));
//    //shared.data[tid] = (shared.data[tid] & (0x55555555 << (tid & 1))) | (((shared.data[tid ^ 1] & (0x55555555 << (tid & 1))) << (~tid & 1)) >> (tid & 1));
//    //shared.data[tid] = __popc(shared.data[tid]);
//
//    barrier(CLK_LOCAL_MEM_FENCE);
//    shared.data[tid + (tid / psize)] = s;
//    //shared.data[tid] = s;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    s = (psize - shared.task.data.residualOrder * (get_local_id(0) + get_group_id(0) == 0)) * (get_local_id(1) + 1);
//    int dpos = __mul24(get_local_id(0), psize + 1);
//    //int dpos = __mul24(get_local_id(0), psize);
//    // calc number of unary bits for part get_local_id(0) with rice paramater get_local_id(1)
//#pragma unroll 0
//    for (int i = 0; i < psize; i++)
//	s += shared.data[dpos + i] >> get_local_id(1);
//
//    // output length
//    const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (get_local_id(1) << (max_porder + 1));
//    if (get_local_id(1) <= 14 && get_local_id(0) < parts)
//	partition_lengths[pos + get_group_id(0) * parts_per_block + get_local_id(0)] = s;
//}
//
//__kernel void cudaCalcPartition16(
//    int* partition_lengths,
//    int* residual,
//    int* samples,
//    FLACCLSubframeTask *tasks,
//    int max_porder, // <= 8
//    int psize, // == 16
//    int parts_per_block // == 16
//    )
//{
//    __local struct {
//	int data[256+32];
//	FLACCLSubframeTask task;
//    } shared;
//    const int tid = get_local_id(0) + (get_local_id(1) << 4);
//    if (tid < sizeof(shared.task) / sizeof(int))
//	((int*)&shared.task)[tid] = ((int*)(&tasks[get_group_id(1)]))[tid];
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    const int offs = (get_group_id(0) << 8) + tid;
//
//    // fetch samples
//    if (tid < 32) shared.data[tid] = min(offs, tid + shared.task.data.residualOrder) >= 32 ? samples[shared.task.data.samplesOffs + offs - 32] >> shared.task.data.wbits : 0;
//    shared.data[32 + tid] = samples[shared.task.data.samplesOffs + offs] >> shared.task.data.wbits;
// //   if (tid < 32 && tid >= shared.task.data.residualOrder)
//	//shared.task.coefs[tid] = 0;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    // compute residual
//    int s = 0;
//    for (int c = -shared.task.data.residualOrder; c < 0; c++)
//	s += __mul24(shared.data[32 + tid + c], shared.task.coefs[shared.task.data.residualOrder + c]);
// //   int spos = 32 + tid - shared.task.data.residualOrder;
// //   int s=
//	//__mul24(shared.data[spos + 0], shared.task.coefs[0]) + __mul24(shared.data[spos + 1], shared.task.coefs[1]) + 
//	//__mul24(shared.data[spos + 2], shared.task.coefs[2]) + __mul24(shared.data[spos + 3], shared.task.coefs[3]) + 
//	//__mul24(shared.data[spos + 4], shared.task.coefs[4]) + __mul24(shared.data[spos + 5], shared.task.coefs[5]) + 
//	//__mul24(shared.data[spos + 6], shared.task.coefs[6]) + __mul24(shared.data[spos + 7], shared.task.coefs[7]) +
//	//__mul24(shared.data[spos + 8], shared.task.coefs[8]) + __mul24(shared.data[spos + 9], shared.task.coefs[9]) + 
//	//__mul24(shared.data[spos + 10], shared.task.coefs[10]) + __mul24(shared.data[spos + 11], shared.task.coefs[11]) +
//	//__mul24(shared.data[spos + 12], shared.task.coefs[12]) + __mul24(shared.data[spos + 13], shared.task.coefs[13]) + 
//	//__mul24(shared.data[spos + 14], shared.task.coefs[14]) + __mul24(shared.data[spos + 15], shared.task.coefs[15]);
//    s = shared.data[32 + tid] - (s >> shared.task.data.shift);
//
//    if (get_group_id(0) != 0 || tid >= shared.task.data.residualOrder)
//	residual[shared.task.data.residualOffs + (get_group_id(0) << 8) + tid] = s;
//    else
//	s = 0;
//
//    // convert to unsigned
//    s = min(0xfffff, (s << 1) ^ (s >> 31));
//    barrier(CLK_LOCAL_MEM_FENCE);
//    shared.data[tid + get_local_id(1)] = s;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    // calc number of unary bits for part get_local_id(0) with rice paramater get_local_id(1)
//    int dpos = __mul24(get_local_id(0), 17);
//    int sum =
//	(shared.data[dpos + 0] >> get_local_id(1)) + (shared.data[dpos + 1] >> get_local_id(1)) + 
//	(shared.data[dpos + 2] >> get_local_id(1)) + (shared.data[dpos + 3] >> get_local_id(1)) + 
//	(shared.data[dpos + 4] >> get_local_id(1)) + (shared.data[dpos + 5] >> get_local_id(1)) + 
//	(shared.data[dpos + 6] >> get_local_id(1)) + (shared.data[dpos + 7] >> get_local_id(1)) + 
//	(shared.data[dpos + 8] >> get_local_id(1)) + (shared.data[dpos + 9] >> get_local_id(1)) + 
//	(shared.data[dpos + 10] >> get_local_id(1)) + (shared.data[dpos + 11] >> get_local_id(1)) + 
//	(shared.data[dpos + 12] >> get_local_id(1)) + (shared.data[dpos + 13] >> get_local_id(1)) + 
//	(shared.data[dpos + 14] >> get_local_id(1)) + (shared.data[dpos + 15] >> get_local_id(1));
//
//    // output length
//    const int pos = ((15 * get_group_id(1) + get_local_id(1)) << (max_porder + 1)) + (get_group_id(0) << 4) + get_local_id(0);
//    if (get_local_id(1) <= 14)
//	partition_lengths[pos] = sum + (16 - shared.task.data.residualOrder * (get_local_id(0) + get_group_id(0) == 0)) * (get_local_id(1) + 1);
//}
//
//__kernel void cudaCalcLargePartition(
//    int* partition_lengths,
//    int* residual,
//    int* samples,
//    FLACCLSubframeTask *tasks,
//    int max_porder, // <= 8
//    int psize, // == >= 128
//    int parts_per_block // == 1
//    )
//{
//    __local struct {
//	int data[256];
//	volatile int length[256];
//	FLACCLSubframeTask task;
//    } shared;
//    const int tid = get_local_id(0) + (get_local_id(1) << 4);
//    if (tid < sizeof(shared.task) / sizeof(int))
//	((int*)&shared.task)[tid] = ((int*)(&tasks[get_group_id(1)]))[tid];
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    int sum = 0;
//    for (int pos = 0; pos < psize; pos += 256)
//    {
//	// fetch residual
//	int offs = get_group_id(0) * psize + pos + tid;
//	int s = (offs >= shared.task.data.residualOrder && pos + tid < psize) ? residual[shared.task.data.residualOffs + offs] : 0;
//	// convert to unsigned
//	shared.data[tid] = min(0xfffff, (s << 1) ^ (s >> 31));
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	// calc number of unary bits for each residual sample with each rice paramater
//#pragma unroll 0
//	for (int i = get_local_id(0); i < min(psize,256); i += 16)
//	    // for sample (i + get_local_id(0)) with this rice paramater (get_local_id(1))
//	    sum += shared.data[i] >> get_local_id(1);
//	barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    shared.length[tid] = min(0xfffff,sum);
//    SUM16(shared.length,tid,+=);
//
//    // output length
//    const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (get_local_id(1) << (max_porder + 1));
//    if (get_local_id(1) <= 14 && get_local_id(0) == 0)
//	partition_lengths[pos + get_group_id(0)] = min(0xfffff,shared.length[tid]) + (psize - shared.task.data.residualOrder * (get_group_id(0) == 0)) * (get_local_id(1) + 1);
//}
//
//// Sums partition lengths for a certain k == get_group_id(0)
//// Requires 128 threads
//__kernel void cudaSumPartition(
//    int* partition_lengths,
//    int max_porder
//    )
//{
//    __local struct {
//	volatile int data[512+32]; // max_porder <= 8, data length <= 1 << 9.
//    } shared;
//
//    const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (get_group_id(0) << (max_porder + 1));
//
//    // fetch partition lengths
//    shared.data[get_local_id(0)] = get_local_id(0) < (1 << max_porder) ? partition_lengths[pos + get_local_id(0)] : 0;
//    shared.data[get_local_size(0) + get_local_id(0)] = get_local_size(0) + get_local_id(0) < (1 << max_porder) ? partition_lengths[pos + get_local_size(0) + get_local_id(0)] : 0;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    int in_pos = (get_local_id(0) << 1);
//    int out_pos = (1 << max_porder) + get_local_id(0);
//    int bs;
//    for (bs = 1 << (max_porder - 1); bs > 32; bs >>= 1)
//    {
//	if (get_local_id(0) < bs) shared.data[out_pos] = shared.data[in_pos] + shared.data[in_pos + 1];
//	in_pos += bs << 1;
//	out_pos += bs;
//	barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    if (get_local_id(0) < 32)
//    for (; bs > 0; bs >>= 1)
//    {
//	shared.data[out_pos] = shared.data[in_pos] + shared.data[in_pos + 1];
//	in_pos += bs << 1;
//	out_pos += bs;
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//    if (get_local_id(0) < (1 << max_porder))
//	partition_lengths[pos + (1 << max_porder) + get_local_id(0)] = shared.data[(1 << max_porder) + get_local_id(0)];
//    if (get_local_size(0) + get_local_id(0) < (1 << max_porder))
//	partition_lengths[pos + (1 << max_porder) + get_local_size(0) + get_local_id(0)] = shared.data[(1 << max_porder) + get_local_size(0) + get_local_id(0)];
//}
//
//// Finds optimal rice parameter for up to 16 partitions at a time.
//// Requires 16x16 threads
//__kernel void cudaFindRiceParameter(
//    int* rice_parameters,
//    int* partition_lengths,
//    int max_porder
//    )
//{
//    __local struct {
//	volatile int length[256];
//	volatile int index[256];
//    } shared;
//    const int tid = get_local_id(0) + (get_local_id(1) << 5);
//    const int parts = min(32, 2 << max_porder);
//    const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (get_local_id(1) << (max_porder + 1));
//
//    // read length for 32 partitions
//    int l1 = (get_local_id(0) < parts) ? partition_lengths[pos + get_group_id(0) * 32 + get_local_id(0)] : 0xffffff;
//    int l2 = (get_local_id(1) + 8 <= 14 && get_local_id(0) < parts) ? partition_lengths[pos + (8 << (max_porder + 1)) + get_group_id(0) * 32 + get_local_id(0)] : 0xffffff;
//    // find best rice parameter
//    shared.index[tid] = get_local_id(1) + ((l2 < l1) << 3);
//    shared.length[tid] = l1 = min(l1, l2);
//    barrier(CLK_LOCAL_MEM_FENCE);
//#pragma unroll 3
//    for (int sh = 7; sh >= 5; sh --)
//    {
//	if (tid < (1 << sh))
//	{
//	    l2 = shared.length[tid + (1 << sh)];
//	    shared.index[tid] = shared.index[tid + ((l2 < l1) << sh)];
//	    shared.length[tid] = l1 = min(l1, l2);
//	}    
//	barrier(CLK_LOCAL_MEM_FENCE);
//    }
//    if (tid < parts)
//    {
//	// output rice parameter
//	rice_parameters[(get_group_id(1) << (max_porder + 2)) + get_group_id(0) * parts + tid] = shared.index[tid];
//	// output length
//	rice_parameters[(get_group_id(1) << (max_porder + 2)) + (1 << (max_porder + 1)) + get_group_id(0) * parts + tid] = shared.length[tid];
//    }
//}
//
//__kernel void cudaFindPartitionOrder(
//    int* best_rice_parameters,
//    FLACCLSubframeTask *tasks,
//    int* rice_parameters,
//    int max_porder
//    )
//{
//    __local struct {
//	int data[512];
//	volatile int tmp[256];
//	int length[32];
//	int index[32];
//	//char4 ch[64];
//	FLACCLSubframeTask task;
//    } shared;
//    const int pos = (get_group_id(1) << (max_porder + 2)) + (2 << max_porder);
//    if (get_local_id(0) < sizeof(shared.task) / sizeof(int))
//	((int*)&shared.task)[get_local_id(0)] = ((int*)(&tasks[get_group_id(1)]))[get_local_id(0)];
//    // fetch partition lengths
//    shared.data[get_local_id(0)] = get_local_id(0) < (2 << max_porder) ? rice_parameters[pos + get_local_id(0)] : 0;
//    shared.data[get_local_id(0) + 256] = get_local_id(0) + 256 < (2 << max_porder) ? rice_parameters[pos + 256 + get_local_id(0)] : 0;
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for (int porder = max_porder; porder >= 0; porder--)
//    {
//	shared.tmp[get_local_id(0)] = (get_local_id(0) < (1 << porder)) * shared.data[(2 << max_porder) - (2 << porder) + get_local_id(0)];
//	barrier(CLK_LOCAL_MEM_FENCE);
//	SUM256(shared.tmp, get_local_id(0), +=);
//	if (get_local_id(0) == 0)
//	    shared.length[porder] = shared.tmp[0] + (4 << porder);
//	barrier(CLK_LOCAL_MEM_FENCE);
//    }
//
//    if (get_local_id(0) < 32)
//    {
//	shared.index[get_local_id(0)] = get_local_id(0);
//	if (get_local_id(0) > max_porder)
//	    shared.length[get_local_id(0)] = 0xfffffff;
//	int l1 = shared.length[get_local_id(0)];
//    #pragma unroll 4
//	for (int sh = 3; sh >= 0; sh --)
//	{
//	    int l2 = shared.length[get_local_id(0) + (1 << sh)];
//	    shared.index[get_local_id(0)] = shared.index[get_local_id(0) + ((l2 < l1) << sh)];
//	    shared.length[get_local_id(0)] = l1 = min(l1, l2);
//	}
//	if (get_local_id(0) == 0)
//	    tasks[get_group_id(1)].data.porder = shared.index[0];
//	if (get_local_id(0) == 0)
//	{
//	    int obits = shared.task.data.obits - shared.task.data.wbits;	    
//	    tasks[get_group_id(1)].data.size =
//		shared.task.data.type == Fixed ? shared.task.data.residualOrder * obits + 6 + l1 :
//		shared.task.data.type == LPC ? shared.task.data.residualOrder * obits + 6 + l1 + 4 + 5 + shared.task.data.residualOrder * shared.task.data.cbits :
//		shared.task.data.type == Constant ? obits : obits * shared.task.data.blocksize;
//	}
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//    int porder = shared.index[0];
//    if (get_local_id(0) < (1 << porder))
//	best_rice_parameters[(get_group_id(1) << max_porder) + get_local_id(0)] = rice_parameters[pos - (2 << porder) + get_local_id(0)];
//    // FIXME: should be bytes?
// //   if (get_local_id(0) < (1 << porder))
//	//shared.tmp[get_local_id(0)] = rice_parameters[pos - (2 << porder) + get_local_id(0)];
// //   barrier(CLK_LOCAL_MEM_FENCE);
// //   if (get_local_id(0) < max(1, (1 << porder) >> 2))
// //   {
//	//char4 ch;
//	//ch.x = shared.tmp[(get_local_id(0) << 2)];
//	//ch.y = shared.tmp[(get_local_id(0) << 2) + 1];
//	//ch.z = shared.tmp[(get_local_id(0) << 2) + 2];
//	//ch.w = shared.tmp[(get_local_id(0) << 2) + 3];
//	//shared.ch[get_local_id(0)] = ch
// //   }	
// //   barrier(CLK_LOCAL_MEM_FENCE);
// //   if (get_local_id(0) < max(1, (1 << porder) >> 2))
//	//best_rice_parameters[(get_group_id(1) << max_porder) + get_local_id(0)] = shared.ch[get_local_id(0)];
//}
//
//#endif
//
//#if 0
//    if (get_local_id(0) < order)
//    {
//	for (int i = 0; i < order; i++)
//	    if (get_local_id(0) >= i)
//		sum[get_local_id(0) - i] += coefs[get_local_id(0)] * sample[order - i - 1];
//	fot (int i = order; i < blocksize; i++)
//	{
//	    if (!get_local_id(0)) sample[order + i] = s = residual[order + i] + (sum[order + i] >> shift);
//	    sum[get_local_id(0) + i + 1] += coefs[get_local_id(0)] * s;
//	}
//    }
//#endif
#endif
