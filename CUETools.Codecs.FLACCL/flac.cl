/**
 * CUETools.FLACCL: FLAC audio encoder using OpenCL
 * Copyright (c) 2010 Gregory S. Chudov
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

#if defined(__Cedar__) || defined(__Redwood__) || defined(__Juniper__) || defined(__Cypress__)
#define AMD
#ifdef DEBUG
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define iclamp(a,b,c) clamp(a,b,c)
#else
#define iclamp(a,b,c) max(b,min(a,c))
#endif

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable

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
    int coefs[32]; // fixme: should be short?
} FLACCLSubframeTask;

__kernel void clStereoDecorr(
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

__kernel void clChannelDecorr2(
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

//__kernel void clChannelDecorr(
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

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clFindWastedBits(
    __global FLACCLSubframeTask *tasks,
    __global int *samples,
    int tasksPerChannel
)
{
    __local int abits[GROUP_SIZE];
    __local int wbits[GROUP_SIZE];
    __local FLACCLSubframeData task;

    int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0) * tasksPerChannel].data))[tid];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    int w = 0, a = 0;
    for (int pos = tid; pos < task.blocksize; pos += GROUP_SIZE)
    {
	int smp = samples[task.samplesOffs + pos];
	w |= smp;
	a |= smp ^ (smp >> 31);
    }
    wbits[tid] = w;
    abits[tid] = a;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = GROUP_SIZE / 2; s > 0; s >>= 1)
    {
	if (tid < s)
	{
	    wbits[tid] |= wbits[tid + s];
	    abits[tid] |= abits[tid + s];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    w = max(0,__ffs(wbits[0]) - 1);
    a = 32 - clz(abits[0]) - w;
    if (tid < tasksPerChannel)
	tasks[get_group_id(0) * tasksPerChannel + tid].data.wbits = w;
    if (tid < tasksPerChannel)
	tasks[get_group_id(0) * tasksPerChannel + tid].data.abits = a;
}

// get_num_groups(0) == number of tasks
// get_num_groups(1) == number of windows
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clComputeAutocor(
    __global float *output,
    __global const int *samples,
    __global const float *window,
    __global FLACCLSubframeTask *tasks,
    const int taskCount // tasks per block
)
{
    __local float data[GROUP_SIZE * 2];
    __local FLACCLSubframeData task;
    const int tid = get_local_id(0);
    // fetch task data
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(tasks + taskCount * get_group_id(0)))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int bs = task.blocksize;
    int windowOffs = get_group_id(1) * bs;

    data[tid] = tid < bs ? samples[task.samplesOffs + tid] * window[windowOffs + tid] : 0.0f;

    const int THREADS_FOR_ORDERS = MAX_ORDER < 8 ? 8 : MAX_ORDER < 16 ? 16 : MAX_ORDER < 32 ? 32 : 64;
    float corr = 0.0f;
    float corr1 = 0.0f;
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	// fetch samples
	float nextData = pos + tid + GROUP_SIZE < bs ? samples[task.samplesOffs + pos + tid + GROUP_SIZE] * window[windowOffs + pos + tid + GROUP_SIZE] : 0.0f;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

#ifdef XXXAMD
	__local float * dptr = &data[tid & ~(THREADS_FOR_ORDERS - 1)];
	float4 res = 0.0f;
	for (int i = 0; i < THREADS_FOR_ORDERS / 4; i++)
	    res += vload4(i, dptr) * vload4(i, &data[tid]);
	corr += res.x + res.y + res.w + res.z;
#else
	int tid1 = tid & ~(THREADS_FOR_ORDERS - 1);
	float res = 0.0f;
	for (int i = 0; i < THREADS_FOR_ORDERS; i++)
	    res += data[tid1 + i] * data[tid + i];
	corr += res;
#endif
	if (THREADS_FOR_ORDERS > 8 && (pos & (GROUP_SIZE * 7)) == 0)
	{
	    corr1 += corr;
	    corr = 0.0f;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
    }

    data[tid] = corr + corr1;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = GROUP_SIZE / 2; i >= THREADS_FOR_ORDERS; i >>= 1)
    {
	if (tid < i)
	    data[tid] += data[tid + i];
	barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid <= MAX_ORDER)
	output[(get_group_id(0) * get_num_groups(1) + get_group_id(1)) * (MAX_ORDER + 1) + tid] = data[tid];
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void clComputeLPC(
    __global float *autoc,
    __global float *lpcs,
    int windowCount
)
{
    __local struct {
	volatile float ldr[32];
	volatile float gen1[32];
	volatile float error[32];
	volatile float autoc[33];
    } shared;
    const int tid = get_local_id(0);// + get_local_id(1) * 32;    
    int autocOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (MAX_ORDER + 1);
    int lpcOffs = autocOffs * 32;
    
    if (get_local_id(0) <= MAX_ORDER)
	shared.autoc[get_local_id(0)] = autoc[autocOffs + get_local_id(0)];
    if (get_local_id(0) + get_local_size(0) <= MAX_ORDER)
	shared.autoc[get_local_id(0) + get_local_size(0)] = autoc[autocOffs + get_local_id(0) + get_local_size(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute LPC using Schur and Levinson-Durbin recursion
    float gen0 = shared.gen1[get_local_id(0)] = shared.autoc[get_local_id(0)+1];
    shared.ldr[get_local_id(0)] = 0.0f;
    float error = shared.autoc[0];
    
#ifdef DEBUGPRINT1
    int magic = shared.autoc[0] == 177286873088.0f;
    if (magic && get_local_id(0) <= MAX_ORDER)
        printf("autoc[%d] == %f\n", get_local_id(0), shared.autoc[get_local_id(0)]);
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int order = 0; order < MAX_ORDER; order++)
    {
	// Schur recursion
	float reff = -shared.gen1[0] / error;
	//error += shared.gen1[0] * reff; // Equivalent to error *= (1 - reff * reff);
	error *= (1 - reff * reff);
	float gen1;
	if (get_local_id(0) < MAX_ORDER - 1 - order)
	{
	    gen1 = shared.gen1[get_local_id(0) + 1] + reff * gen0;
	    gen0 += shared.gen1[get_local_id(0) + 1] * reff;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) < MAX_ORDER - 1 - order)
	    shared.gen1[get_local_id(0)] = gen1;
#ifdef DEBUGPRINT1
	if (magic && get_local_id(0) == 0)
	    printf("order == %d, reff == %f, error = %f\n", order, reff, error);
	if (magic && get_local_id(0) <= MAX_ORDER)
	    printf("gen[%d] == %f, %f\n", get_local_id(0), gen0, gen1);
#endif

	// Store prediction error
	if (get_local_id(0) == 0)
	    shared.error[order] = error;

	// Levinson-Durbin recursion
	float ldr = shared.ldr[get_local_id(0)];
        barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) < order)
	    shared.ldr[order - 1 - get_local_id(0)] += reff * ldr;
	if (get_local_id(0) == order)
	    shared.ldr[get_local_id(0)] += reff;
        barrier(CLK_LOCAL_MEM_FENCE);

	// Output coeffs
	if (get_local_id(0) <= order)
	    lpcs[lpcOffs + order * 32 + get_local_id(0)] = -shared.ldr[order - get_local_id(0)];
	//if (get_local_id(0) <= order + 1 && fabs(-shared.ldr[0]) > 3000)
	//    printf("coef[%d] == %f, autoc == %f, error == %f\n", get_local_id(0), -shared.ldr[order - get_local_id(0)], shared.autoc[get_local_id(0)], shared.error[get_local_id(0)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Output prediction error estimates
    if (get_local_id(0) < MAX_ORDER)
	lpcs[lpcOffs + MAX_ORDER * 32 + get_local_id(0)] = shared.error[get_local_id(0)];
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void clQuantizeLPC(
    __global FLACCLSubframeTask *tasks,
    __global float*lpcs,
    int taskCount, // tasks per block
    int taskCountLPC, // tasks per set of coeffs (<= 32)
    int minprecision,
    int precisions
)
{
    __local struct {
	FLACCLSubframeData task;
	volatile int index[64];
	volatile float error[64];
	volatile int maxcoef[32];
	volatile int maxcoef2[32];
    } shared;

    const int tid = get_local_id(0);

    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((__local int*)&shared.task)[tid] = ((__global int*)(tasks + get_group_id(1) * taskCount))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    const int lpcOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (MAX_ORDER + 1) * 32;

    // Select best orders based on Akaike's Criteria
    shared.index[tid] = min(MAX_ORDER - 1, tid);
    shared.error[tid] = shared.task.blocksize * 64 + tid;
    shared.index[32 + tid] = MAX_ORDER - 1;
    shared.error[32 + tid] = shared.task.blocksize * 64 + tid + 32;
    shared.maxcoef[tid] = 0;
    shared.maxcoef2[tid] = 0;

    // Load prediction error estimates
    if (tid < MAX_ORDER)
	shared.error[tid] = shared.task.blocksize * log(lpcs[lpcOffs + MAX_ORDER * 32 + tid]) + tid * 4.12f * log((float)shared.task.blocksize);
	//shared.error[get_local_id(0)] = shared.task.blocksize * log(lpcs[lpcOffs + MAX_ORDER * 32 + get_local_id(0)]) + get_local_id(0) * 0.30f * (shared.task.abits + 1) * log(shared.task.blocksize);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sort using bitonic sort
    for(int size = 2; size < 64; size <<= 1){
	//Bitonic merge
	int ddd = (tid & (size / 2)) == 0;
	for(int stride = size / 2; stride > 0; stride >>= 1){
	    int pos = 2 * tid - (tid & (stride - 1));
	    float e0 = shared.error[pos];
	    float e1 = shared.error[pos + stride];
	    int i0 = shared.index[pos];
	    int i1 = shared.index[pos + stride];
	    barrier(CLK_LOCAL_MEM_FENCE);
	    if ((e0 >= e1) == ddd)
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
	    int pos = 2 * tid - (tid & (stride - 1));
	    float e0 = shared.error[pos];
	    float e1 = shared.error[pos + stride];
	    int i0 = shared.index[pos];
	    int i1 = shared.index[pos + stride];
	    barrier(CLK_LOCAL_MEM_FENCE);
	    if (e0 >= e1)
	    {
		shared.error[pos] = e1;
		shared.error[pos + stride] = e0;
		shared.index[pos] = i1;
		shared.index[pos + stride] = i0;
	    }
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
    }

    //shared.index[tid] = MAX_ORDER - 1;
    //barrier(CLK_LOCAL_MEM_FENCE);

    // Quantization
    for (int i = 0; i < taskCountLPC; i ++)
    {
	int order = shared.index[i >> precisions];
	float lpc = tid <= order ? lpcs[lpcOffs + order * 32 + tid] : 0.0f;
	// get 15 bits of each coeff
	int coef = convert_int_rte(lpc * (1 << 15));
	// remove sign bits
	atom_or(shared.maxcoef + i, coef ^ (coef >> 31));
	barrier(CLK_LOCAL_MEM_FENCE);
	//SUM32(shared.tmpi,tid,|=);
	// choose precision	
	//int cbits = max(3, min(10, 5 + (shared.task.abits >> 1))); //  - convert_int_rte(shared.PE[order - 1])
	int cbits = max(3, min(min(13 - minprecision + (i - ((i >> precisions) << precisions)) - (shared.task.blocksize <= 2304) - (shared.task.blocksize <= 1152) - (shared.task.blocksize <= 576), shared.task.abits), clz(order) + 1 - shared.task.abits));
	// calculate shift based on precision and number of leading zeroes in coeffs
	int shift = max(0,min(15, clz(shared.maxcoef[i]) - 18 + cbits));

	//cbits = 13;
	//shift = 15;

	//if (shared.task.abits + 32 - clz(order) < shift
	//int shift = max(0,min(15, (shared.task.abits >> 2) - 14 + clz(shared.tmpi[tid & ~31]) + ((32 - clz(order))>>1)));
	// quantize coeffs with given shift
	coef = convert_int_rte(clamp(lpc * (1 << shift), (float)(-1 << (cbits - 1)), (float)(1 << (cbits - 1))));
	// error correction
	//shared.tmp[tid] = (tid != 0) * (shared.arp[tid - 1]*(1 << shared.task.shift) - shared.task.coefs[tid - 1]);
	//shared.task.coefs[tid] = max(-(1 << (shared.task.cbits - 1)), min((1 << (shared.task.cbits - 1))-1, convert_int_rte((shared.arp[tid]) * (1 << shared.task.shift) + shared.tmp[tid])));
	// remove sign bits
	atom_or(shared.maxcoef2 + i, coef ^ (coef >> 31));
	barrier(CLK_LOCAL_MEM_FENCE);
	// calculate actual number of bits (+1 for sign)
	cbits = 1 + 32 - clz(shared.maxcoef2[i]);

	// output shift, cbits and output coeffs
	int taskNo = get_group_id(1) * taskCount + get_group_id(0) * taskCountLPC + i;
	if (tid == 0)
	    tasks[taskNo].data.shift = shift;
	if (tid == 0)
	    tasks[taskNo].data.cbits = cbits;
	if (tid == 0)
	    tasks[taskNo].data.residualOrder = order + 1;
	if (tid <= order)
	    tasks[taskNo].coefs[tid] = coef;
    }
}

__kernel /*__attribute__(( vec_type_hint (int4)))*/ __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clEstimateResidual(
    __global int*samples,
    __global FLACCLSubframeTask *tasks
    )
{
    __local float data[GROUP_SIZE * 2];
    __local FLACCLSubframeTask task;
    __local int psum[64];
    __local float fcoef[32];

    const int tid = get_local_id(0);
    if (tid < sizeof(task)/sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0)]))[tid];
    barrier(CLK_GLOBAL_MEM_FENCE);

    int ro = task.data.residualOrder;
    int bs = task.data.blocksize;

    if (tid < 32)
	fcoef[tid] = select(0.0f, - ((float) task.coefs[tid]) / (1 << task.data.shift), tid < ro);
	//fcoef[tid] = select(0.0f, - ((float) task.coefs[tid + ro - MAX_ORDER]) / (1 << task.data.shift), tid + ro >= MAX_ORDER && tid < MAX_ORDER);
    if (tid < 64)
	psum[tid] = 0;
    data[tid] = 0.0f;

    int partOrder = max(1, clz(64) - clz(bs - 1) + 1);

    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef AMD
    float4 cptr0 = vload4(0, &fcoef[0]);
    float4 cptr1 = vload4(1, &fcoef[0]);
#if MAX_ORDER > 8
    float4 cptr2 = vload4(2, &fcoef[0]);
#endif
#endif
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	// fetch samples
	int offs = pos + tid;
	float nextData = offs < bs ? samples[task.data.samplesOffs + offs] >> task.data.wbits : 0.0f;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	// compute residual
	__local float* dptr = &data[tid + GROUP_SIZE - ro];
#ifdef AMD
	float4 sum = cptr0 * vload4(0, dptr)
	    + cptr1 * vload4(1, dptr)
#if MAX_ORDER > 8
	    + cptr2 * vload4(2, dptr)
  #if MAX_ORDER > 12
	    + vload4(3, &fcoef[0]) * vload4(3, dptr)
    #if MAX_ORDER > 16
	    + vload4(4, &fcoef[0]) * vload4(4, dptr)
	    + vload4(5, &fcoef[0]) * vload4(5, dptr)
	    + vload4(6, &fcoef[0]) * vload4(6, dptr)
	    + vload4(7, &fcoef[0]) * vload4(7, dptr)
    #endif
  #endif
#endif
	    ;

	int t = convert_int_rte(nextData + sum.x + sum.y + sum.z + sum.w);
#else
	float sum = 
	    fcoef[0] * dptr[0] + fcoef[1] * dptr[1] + fcoef[2] * dptr[2] + fcoef[3] * dptr[3] + 
	    fcoef[4] * dptr[4] + fcoef[5] * dptr[5] + fcoef[6] * dptr[6] + fcoef[7] * dptr[7] + 
	    fcoef[8] * dptr[8] + fcoef[9] * dptr[9] + fcoef[10] * dptr[10] + fcoef[11] * dptr[11] ;
	int t = convert_int_rte(nextData + sum);
#endif
	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
	// ensure we're within frame bounds
	t = select(0, t, offs >= ro && offs < bs);
	// overflow protection
	t = iclamp(t, -0x7fffff, 0x7fffff);
	// convert to unsigned
	//if (offs < bs)
	    atom_add(&psum[offs >> partOrder], (t << 1) ^ (t >> 31));
    }

    // calculate rice partition bit length for every (1 << partOrder) samples
    if (tid < 64)
    {
	int k = iclamp(clz(1 << partOrder) - clz(psum[tid]), 0, 14); // 27 - clz(res) == clz(16) - clz(res) == log2(res / 16)
	psum[tid] = (k << partOrder) + (psum[tid] >> k);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = 32; l > 0; l >>= 1)
    {
	if (tid < l)
	    psum[tid] += psum[tid + l];
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
    {
	int pl = psum[0] + (bs - ro);
	int obits = task.data.obits - task.data.wbits;
	int len = min(obits * task.data.blocksize,
		task.data.type == Fixed ? task.data.residualOrder * obits + 6 + (4 * 1/2) + pl :
		task.data.type == LPC ? task.data.residualOrder * obits + 4 + 5 + task.data.residualOrder * task.data.cbits + 6 + (4 * 1/2)/* << porder */ + pl :
		task.data.type == Constant ? obits * select(1, task.data.blocksize, pl != task.data.blocksize - task.data.residualOrder) : 
		obits * task.data.blocksize);
	tasks[get_group_id(0)].data.size = len;
    }
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void clChooseBestMethod(
    __global FLACCLSubframeTask *tasks,
    int taskCount
    )
{
    int best_length = 0x7fffffff;
    int best_index = 0;
    const int tid = get_local_id(0);
    
    for (int taskNo = 0; taskNo < taskCount; taskNo++)
    {
	if (tid == 0)
	{
	    int len = tasks[taskNo + taskCount * get_group_id(0)].data.size;
	    if (len < best_length)
	    {
		best_length = len;
		best_index = taskNo;
	    }
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0)
	tasks[taskCount * get_group_id(0)].data.best_index = taskCount * get_group_id(0) + best_index;
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void clCopyBestMethod(
    __global FLACCLSubframeTask *tasks_out,
    __global FLACCLSubframeTask *tasks,
    int count
    )
{
    __local int best_index;
    if (get_local_id(0) == 0)
	best_index = tasks[count * get_group_id(0)].data.best_index;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < sizeof(FLACCLSubframeTask)/sizeof(int))
	((__global int*)(tasks_out + get_group_id(0)))[get_local_id(0)] = ((__global int*)(tasks + best_index))[get_local_id(0)];
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void clCopyBestMethodStereo(
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
	shared.best_index[get_local_id(0)] = tasks[count * (get_group_id(0) * 4 + get_local_id(0))].data.best_index;
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
	((__global int*)(tasks_out + 2 * get_group_id(0)))[get_local_id(0)] = ((__global int*)(tasks + shared.lr_index[0]))[get_local_id(0)];
    if (get_local_id(0) == 0)
	tasks_out[2 * get_group_id(0)].data.residualOffs = tasks[shared.best_index[0]].data.residualOffs;
    if (get_local_id(0) < sizeof(FLACCLSubframeTask)/sizeof(int))
	((__global int*)(tasks_out + 2 * get_group_id(0) + 1))[get_local_id(0)] = ((__global int*)(tasks + shared.lr_index[1]))[get_local_id(0)];
    if (get_local_id(0) == 0)
	tasks_out[2 * get_group_id(0) + 1].data.residualOffs = tasks[shared.best_index[1]].data.residualOffs;
}

// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clEncodeResidual(
    __global int *output,
    __global int *samples,
    __global FLACCLSubframeTask *tasks
    )
{
    __local FLACCLSubframeTask task;
    __local int data[GROUP_SIZE * 2];
    const int tid = get_local_id(0);
    if (get_local_id(0) < sizeof(task) / sizeof(int))
	((__local int*)&task)[get_local_id(0)] = ((__global int*)(&tasks[get_group_id(0)]))[get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;

    if (tid < 32 && tid >= ro)
	task.coefs[tid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int4 cptr0 = vload4(0, &task.coefs[0]);
#if MAX_ORDER > 4
    int4 cptr1 = vload4(1, &task.coefs[0]);
#if MAX_ORDER > 8
    int4 cptr2 = vload4(2, &task.coefs[0]);
#endif
#endif

    data[tid] = 0;
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	// fetch samples
	int off = pos + tid;
	int nextData = off < bs ? samples[task.data.samplesOffs + off] >> task.data.wbits : 0;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	// compute residual
	__local int* dptr = &data[tid + GROUP_SIZE - ro];
	int4 sum = cptr0 * vload4(0, dptr)
#if MAX_ORDER > 4
	    + cptr1 * vload4(1, dptr)
#if MAX_ORDER > 8
	    + cptr2 * vload4(2, dptr)
#if MAX_ORDER > 12
	    + vload4(3, &task.coefs[0]) * vload4(3, dptr)
#if MAX_ORDER > 16
	    + vload4(4, &task.coefs[0]) * vload4(4, dptr)
	    + vload4(5, &task.coefs[0]) * vload4(5, dptr)
	    + vload4(6, &task.coefs[0]) * vload4(6, dptr)
	    + vload4(7, &task.coefs[0]) * vload4(7, dptr)
#endif
#endif
#endif
#endif
	    ;
	if (off >= ro && off < bs)
	    output[task.data.residualOffs + off] = data[tid + GROUP_SIZE] - ((sum.x + sum.y + sum.z + sum.w) >> task.data.shift);

	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
    }
}

// get_group_id(0) == partition index / (GROUP_SIZE / 16)
// get_group_id(1) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clCalcPartition(
    __global int *partition_lengths,
    __global int *residual,
    __global FLACCLSubframeTask *tasks,
    int max_porder, // <= 8
    int psize // == task.blocksize >> max_porder?
    )
{
    __local int pl[(GROUP_SIZE / 8)][15];
    __local FLACCLSubframeData task;

    const int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(1)]))[tid];
    if (tid < (GROUP_SIZE / 8))
    {
	for (int k = 0; k <= 14; k++)
	    pl[tid][k] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int start = get_group_id(0) * psize * (GROUP_SIZE / 16);
    int end = min(start + psize * (GROUP_SIZE / 16), task.blocksize);
    for (int offs = start + tid; offs < end; offs += GROUP_SIZE)
    {
	// fetch residual
	int s = (offs >= task.residualOrder && offs < end) ? residual[task.residualOffs + offs] : 0;
	// overflow protection
	s = iclamp(s, -0x7fffff, 0x7fffff);
	// convert to unsigned
	s = (s << 1) ^ (s >> 31);
	// calc number of unary bits for each residual sample with each rice paramater
	int part = (offs - start) / psize + (tid & 1) * (GROUP_SIZE / 16);
	for (int k = 0; k <= 14; k++)
	    atom_add(&pl[part][k], s >> k);
	    //pl[part][k] += s >> k;
    }   
    barrier(CLK_LOCAL_MEM_FENCE);

    int part = get_group_id(0) * (GROUP_SIZE / 16) + tid;
    if (tid < (GROUP_SIZE / 16) && part < (1 << max_porder))
    {
	for (int k = 0; k <= 14; k++)
	{
	    // output length
	    const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (k << (max_porder + 1));
	    int plen = pl[tid][k] + pl[tid + (GROUP_SIZE / 16)][k];
	    partition_lengths[pos + part] = min(0x7fffff, plen) + (psize - select(0, task.residualOrder, part == 0)) * (k + 1);
	 //   if (get_group_id(1) == 0)
		//printf("pl[%d][%d] == %d\n", k, part, min(0x7fffff, pl[k][tid]) + (psize - task.residualOrder * (part == 0)) * (k + 1));
	}
    }
}

// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clCalcPartition16(
    __global int *partition_lengths,
    __global int *residual,
    __global int *samples,
    __global FLACCLSubframeTask *tasks,
    int max_porder // <= 8
    )
{
    __local FLACCLSubframeTask task;
    __local int data[GROUP_SIZE * 2];
    __local int res[GROUP_SIZE];

    const int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0)]))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;

    if (tid >= ro && tid < 32)
	task.coefs[tid] = 0;

    int k = tid & 15;
    int x = tid / 16;

    barrier(CLK_LOCAL_MEM_FENCE);

    int4 cptr0 = vload4(0, &task.coefs[0]);
    data[tid] = 0;
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	int offs = pos + tid;
	// fetch samples
	int nextData = offs < bs ? samples[task.data.samplesOffs + offs] >> task.data.wbits : 0;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// compute residual
	__local int* dptr = &data[tid + GROUP_SIZE - ro];
	int4 sum = cptr0 * vload4(0, dptr)
#if MAX_ORDER > 4
	    + vload4(1, &task.coefs[0]) * vload4(1, dptr)
#if MAX_ORDER > 8
	    + vload4(2, &task.coefs[0]) * vload4(2, dptr)
#if MAX_ORDER > 12
	    + vload4(3, &task.coefs[0]) * vload4(3, dptr)
#if MAX_ORDER > 16
	    + vload4(4, &task.coefs[0]) * vload4(4, dptr)
	    + vload4(5, &task.coefs[0]) * vload4(5, dptr)
	    + vload4(6, &task.coefs[0]) * vload4(6, dptr)
	    + vload4(7, &task.coefs[0]) * vload4(7, dptr)
#endif
#endif
#endif
#endif
	    ;
	int s = select(0, nextData - ((sum.x + sum.y + sum.z + sum.w) >> task.data.shift), offs >= ro && offs < bs);

	// output residual
	if (offs < bs)
	    residual[task.data.residualOffs + offs] = s;

	//int s = select(0, residual[task.data.residualOffs + offs], offs >= ro && offs < bs);
	
	s = iclamp(s, -0x7fffff, 0x7fffff);
	// convert to unsigned
	res[tid] = (s << 1) ^ (s >> 31);

	// for (int k = 0; k < 15; k++) atom_add(&pl[x][k], s >> k);

	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;

	// calc number of unary bits for each residual sample with each rice paramater
	__local int * chunk = &res[x << 4];
	sum = (vload4(0,chunk) >> k) + (vload4(1,chunk) >> k) + (vload4(2,chunk) >> k) + (vload4(3,chunk) >> k);
	s = sum.x + sum.y + sum.z + sum.w;

	const int lpos = (15 << (max_porder + 1)) * get_group_id(0) + (k << (max_porder + 1)) + offs / 16;
	if (k <= 14)
	    partition_lengths[lpos] = min(0x7fffff, s) + (16 - select(0, ro, offs < 16)) * (k + 1);
    }    
}

// Sums partition lengths for a certain k == get_group_id(0)
// Requires 128 threads
// get_group_id(0) == k
// get_group_id(1) == task index
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void clSumPartition(
    __global int* partition_lengths,
    int max_porder
    )
{
    __local int data[256]; // max_porder <= 8, data length <= 1 << 9.
    const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (get_group_id(0) << (max_porder + 1));

    // fetch partition lengths
    int2 pl = get_local_id(0) * 2 < (1 << max_porder) ? vload2(get_local_id(0),&partition_lengths[pos]) : 0;
    data[get_local_id(0)] = pl.x + pl.y;
    barrier(CLK_LOCAL_MEM_FENCE);

    int in_pos = (get_local_id(0) << 1);
    int out_pos = (1 << (max_porder - 1)) + get_local_id(0);
    for (int bs = 1 << (max_porder - 2); bs > 0; bs >>= 1)
    {
	if (get_local_id(0) < bs) data[out_pos] = data[in_pos] + data[in_pos + 1];
	in_pos += bs << 1;
	out_pos += bs;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) < (1 << max_porder))
	partition_lengths[pos + (1 << max_porder) + get_local_id(0)] = data[get_local_id(0)];
    if (get_local_size(0) + get_local_id(0) < (1 << max_porder))
	partition_lengths[pos + (1 << max_porder) + get_local_size(0) + get_local_id(0)] = data[get_local_size(0) + get_local_id(0)];
}

// Finds optimal rice parameter for each partition.
// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clFindRiceParameter(
    __global FLACCLSubframeTask *tasks,
    __global int* rice_parameters,
    __global int* partition_lengths,
    int max_porder
    )
{
    for (int offs = get_local_id(0); offs < (2 << max_porder); offs += GROUP_SIZE)
    {
	const int pos = (15 << (max_porder + 1)) * get_group_id(0) + offs;
	int best_l = partition_lengths[pos];
	int best_k = 0;
	for (int k = 1; k <= 14; k++)
	{
	    int l = partition_lengths[pos + (k << (max_porder + 1))];
	    best_k = select(best_k, k, l < best_l);
	    best_l = min(best_l, l);
	}

	// output rice parameter
	rice_parameters[(get_group_id(0) << (max_porder + 2)) + offs] = best_k;
	// output length
	rice_parameters[(get_group_id(0) << (max_porder + 2)) + (1 << (max_porder + 1)) + offs] = best_l;
    }
}

// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clFindPartitionOrder(
    __global int *residual,
    __global int* best_rice_parameters,
    __global FLACCLSubframeTask *tasks,
    __global int* rice_parameters,
    int max_porder
    )
{
    __local int partlen[16];
    __local FLACCLSubframeData task;

    const int pos = (get_group_id(0) << (max_porder + 2)) + (2 << max_porder);
    if (get_local_id(0) < sizeof(task) / sizeof(int))
	((__local int*)&task)[get_local_id(0)] = ((__global int*)(&tasks[get_group_id(0)]))[get_local_id(0)];
    if (get_local_id(0) < 16)
	partlen[get_local_id(0)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // fetch partition lengths
    int lim = (2 << max_porder) - 1;
    for (int offs = get_local_id(0); offs < lim; offs += GROUP_SIZE)
    {
	int len = rice_parameters[pos + offs];
	int porder = 31 - clz(lim - offs);
	atom_add(&partlen[porder], len);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int best_length = partlen[0] + 4;
    int best_porder = 0;
    for (int porder = 1; porder <= max_porder; porder++)
    {
	int length = (4 << porder) + partlen[porder];
	best_porder = select(best_porder, porder, length < best_length);
	best_length = min(best_length, length);
    }

    if (get_local_id(0) == 0)
    {
	tasks[get_group_id(0)].data.porder = best_porder;
	int obits = task.obits - task.wbits;
	tasks[get_group_id(0)].data.size =
	    task.type == Fixed ? task.residualOrder * obits + 6 + best_length :
	    task.type == LPC ? task.residualOrder * obits + 6 + best_length + 4 + 5 + task.residualOrder * task.cbits :
	    task.type == Constant ? obits : obits * task.blocksize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int offs = get_local_id(0); offs < (1 << best_porder); offs += GROUP_SIZE)
	best_rice_parameters[(get_group_id(0) << max_porder) + offs] = rice_parameters[pos - (2 << best_porder) + offs];
    // FIXME: should be bytes?
}
#endif
