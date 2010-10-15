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

//#pragma OPENCL EXTENSION cl_amd_fp64 : enable

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

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void cudaFindWastedBits(
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
    for (int pos = 0; pos < task.blocksize; pos += GROUP_SIZE)
    {
	int smp = pos + tid < task.blocksize ? samples[task.samplesOffs + pos + tid] : 0;
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

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void cudaComputeAutocor(
    __global float *output,
    __global const int *samples,
    __global const float *window,
    __global FLACCLSubframeTask *tasks,
    const int windowCount, // windows (log2: 0,1)
    const int taskCount // tasks per block
)
{
    __local float data[GROUP_SIZE * 2];
    __local float product[GROUP_SIZE];
    __local FLACCLSubframeData task;
    const int tid = get_local_id(0);
    // fetch task data
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(tasks + taskCount * (get_group_id(1) >> windowCount)))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int bs = task.blocksize;
    int windowOffs = (get_group_id(1) & ((1 << windowCount)-1)) * bs;

    data[tid] = tid < bs ? samples[task.samplesOffs + tid] * window[windowOffs + tid] : 0.0f;

    int tid0 = tid % (GROUP_SIZE >> 2);
    int tid1 = tid / (GROUP_SIZE >> 2);
    int lag0 = get_group_id(0) * 4;
    __local float4 * dptr = ((__local float4 *)&data[0]) + tid0;
    __local float4 * dptr1 = ((__local float4 *)&data[lag0 + tid1]) + tid0;
    
    float prod = 0.0f;
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	// fetch samples
	float nextData = pos + tid + GROUP_SIZE < bs ? samples[task.samplesOffs + pos + tid + GROUP_SIZE] * window[windowOffs + pos + tid + GROUP_SIZE] : 0.0f;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	prod += dot(*dptr, *dptr1);

	barrier(CLK_LOCAL_MEM_FENCE);

	data[tid] = nextData;
    }
    product[tid] = prod;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = (GROUP_SIZE >> 3); l > 0; l >>= 1)
    {
	if (tid0 < l)
	    product[tid] = product[tid] + product[tid + l];
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < 4 && tid + lag0 <= MAX_ORDER)
	output[get_group_id(1) * (MAX_ORDER + 1) + tid + lag0] = product[tid * (GROUP_SIZE >> 2)];
}

//#define DEBUGPRINT

#ifdef DEBUGPRINT
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void cudaComputeLPC(
    __global FLACCLSubframeTask *tasks,
    __global float *autoc,
    __global float *lpcs,
    int taskCount, // tasks per block
    int windowCount
)
{
    __local struct {
	FLACCLSubframeData task;
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
	((__local int*)&shared.task)[tid] = ((__global int*)(tasks + get_group_id(1)))[tid];
    if (tid == 0)
    {
	shared.lpcOffs = (get_group_id(0) + get_group_id(1) * windowCount) * (MAX_ORDER + 1) * 32;
	shared.autocOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (MAX_ORDER + 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (get_local_id(0) <= MAX_ORDER)
	shared.autoc[get_local_id(0)] = autoc[shared.autocOffs + get_local_id(0)];
    if (get_local_id(0) + get_local_size(0) <= MAX_ORDER)
	shared.autoc[get_local_id(0) + get_local_size(0)] = autoc[shared.autocOffs + get_local_id(0) + get_local_size(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute LPC using Schur and Levinson-Durbin recursion
    float gen0 = shared.gen1[get_local_id(0)] = shared.autoc[get_local_id(0)+1];
    shared.ldr[get_local_id(0)] = 0.0f;
    float error = shared.autoc[0];
    
#ifdef DEBUGPRINT
    int magic = shared.autoc[0] == 177286873088.0f;
    if (magic && get_local_id(0) <= MAX_ORDER)
        printf("autoc[%d] == %f\n", get_local_id(0), shared.autoc[get_local_id(0)]);
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int order = 0; order < MAX_ORDER; order++)
    {
	// Schur recursion
	float reff = -shared.gen1[0] / error;
	error += shared.gen1[0] * reff; // Equivalent to error *= (1 - reff * reff);
	//error *= (1 - reff * reff);
	float gen1;
	if (get_local_id(0) < MAX_ORDER - 1 - order)
	{
	    gen1 = shared.gen1[get_local_id(0) + 1] + reff * gen0;
	    gen0 += shared.gen1[get_local_id(0) + 1] * reff;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) < MAX_ORDER - 1 - order)
	    shared.gen1[get_local_id(0)] = gen1;
#ifdef DEBUGPRINT
	if (magic && get_local_id(0) == 0)
	    printf("order == %d, reff == %f, error = %f\n", order, reff, error);
	if (magic && get_local_id(0) <= MAX_ORDER)
	    printf("gen[%d] == %f, %f\n", get_local_id(0), gen0, gen1);
#endif

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
	//if (get_local_id(0) <= order + 1 && fabs(-shared.ldr[0]) > 3000)
	//    printf("coef[%d] == %f, autoc == %f, error == %f\n", get_local_id(0), -shared.ldr[order - get_local_id(0)], shared.autoc[get_local_id(0)], shared.error[get_local_id(0)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Output prediction error estimates
    if (get_local_id(0) < MAX_ORDER)
	lpcs[shared.lpcOffs + MAX_ORDER * 32 + get_local_id(0)] = shared.error[get_local_id(0)];
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void cudaQuantizeLPC(
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
	volatile int tmpi[32];
	volatile int index[64];
	volatile float error[64];
	volatile int lpcOffs;
    } shared;

    const int tid = get_local_id(0);

    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((__local int*)&shared.task)[tid] = ((__global int*)(tasks + get_group_id(1) * taskCount))[tid];
    if (tid == 0)
	shared.lpcOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (MAX_ORDER + 1) * 32;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Select best orders based on Akaike's Criteria
    shared.index[tid] = min(MAX_ORDER - 1, tid);
    shared.error[tid] = shared.task.blocksize * 64 + tid;
    shared.index[32 + tid] = MAX_ORDER - 1;
    shared.error[32 + tid] = shared.task.blocksize * 64 + tid + 32;

    // Load prediction error estimates
    if (tid < MAX_ORDER)
	shared.error[tid] = shared.task.blocksize * log(lpcs[shared.lpcOffs + MAX_ORDER * 32 + tid]) + tid * 4.12f * log(shared.task.blocksize);
	//shared.error[get_local_id(0)] = shared.task.blocksize * log(lpcs[shared.lpcOffs + MAX_ORDER * 32 + get_local_id(0)]) + get_local_id(0) * 0.30f * (shared.task.abits + 1) * log(shared.task.blocksize);
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
	float lpc = tid <= order ? lpcs[shared.lpcOffs + order * 32 + tid] : 0.0f;
	// get 15 bits of each coeff
	int coef = convert_int_rte(lpc * (1 << 15));
	// remove sign bits
	shared.tmpi[tid] = coef ^ (coef >> 31);
	barrier(CLK_LOCAL_MEM_FENCE);
	// OR reduction
	for (int l = get_local_size(0) / 2; l > 1; l >>= 1)
	{
	    if (tid < l)
		shared.tmpi[tid] |= shared.tmpi[tid + l];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	//SUM32(shared.tmpi,tid,|=);
	// choose precision	
	//int cbits = max(3, min(10, 5 + (shared.task.abits >> 1))); //  - convert_int_rte(shared.PE[order - 1])
	int cbits = max(3, min(min(13 - minprecision + (i - ((i >> precisions) << precisions)) - (shared.task.blocksize <= 2304) - (shared.task.blocksize <= 1152) - (shared.task.blocksize <= 576), shared.task.abits), clz(order) + 1 - shared.task.abits));
	// calculate shift based on precision and number of leading zeroes in coeffs
	int shift = max(0,min(15, clz(shared.tmpi[0] | shared.tmpi[1]) - 18 + cbits));

	//cbits = 13;
	//shift = 15;

	//if (shared.task.abits + 32 - clz(order) < shift
	//int shift = max(0,min(15, (shared.task.abits >> 2) - 14 + clz(shared.tmpi[tid & ~31]) + ((32 - clz(order))>>1)));
	// quantize coeffs with given shift
	coef = convert_int_rte(clamp(lpc * (1 << shift), -1 << (cbits - 1), 1 << (cbits - 1)));
	// error correction
	//shared.tmp[tid] = (tid != 0) * (shared.arp[tid - 1]*(1 << shared.task.shift) - shared.task.coefs[tid - 1]);
	//shared.task.coefs[tid] = max(-(1 << (shared.task.cbits - 1)), min((1 << (shared.task.cbits - 1))-1, convert_int_rte((shared.arp[tid]) * (1 << shared.task.shift) + shared.tmp[tid])));
	// remove sign bits
	shared.tmpi[tid] = coef ^ (coef >> 31);
	barrier(CLK_LOCAL_MEM_FENCE);
	// OR reduction
	for (int l = get_local_size(0) / 2; l > 1; l >>= 1)
	{
	    if (tid < l)
		shared.tmpi[tid] |= shared.tmpi[tid + l];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	//SUM32(shared.tmpi,tid,|=);
	// calculate actual number of bits (+1 for sign)
	cbits = 1 + 32 - clz(shared.tmpi[0] | shared.tmpi[1]);

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

#define BEACCURATE

__kernel /*__attribute__(( vec_type_hint (int4)))*/ __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void cudaEstimateResidual(
    __global int*output,
    __global int*samples,
    __global FLACCLSubframeTask *tasks
    )
{
    __local int data[GROUP_SIZE * 2];
    __local FLACCLSubframeTask task;
#ifdef BEACCURATE
    __local int residual[GROUP_SIZE];
    __local int len[GROUP_SIZE / 16];
#else
    __local float residual[GROUP_SIZE];
#endif

    const int tid = get_local_id(0);
    if (tid < sizeof(task)/sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0)]))[tid];
    barrier(CLK_GLOBAL_MEM_FENCE);

    int ro = task.data.residualOrder;
    int bs = task.data.blocksize;

    if (tid < 32 && tid >= ro)
	task.coefs[tid] = 0;
#ifdef BEACCURATE
    if (tid < GROUP_SIZE / 16)
	len[tid] = 0;
#else
    long res = 0;
#endif
    data[tid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    __local int4 * cptr = (__local int4 *)&task.coefs[0];
    int4 cptr0 = cptr[0];
#if MAX_ORDER > 4
    int4 cptr1 = cptr[1];
#if MAX_ORDER > 8
    int4 cptr2 = cptr[2];
#endif
#endif
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	// fetch samples
	int offs = pos + tid;
	int nextData = offs < bs ? samples[task.data.samplesOffs + offs] >> task.data.wbits : 0;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	// compute residual
	__local int4 * dptr = (__local int4 *)&data[tid + GROUP_SIZE - ro];
	int4 sum = dptr[0] * cptr0
#if MAX_ORDER > 4
	    + dptr[1] * cptr1
#if MAX_ORDER > 8
	    + dptr[2] * cptr2
#if MAX_ORDER > 12
	    + dptr[3] * cptr[3]
#if MAX_ORDER > 16
	    + dptr[4] * cptr[4]
	    + dptr[5] * cptr[5]
	    + dptr[6] * cptr[6]
	    + dptr[7] * cptr[7]
#endif
#endif
#endif
#endif
	    ;
	
	int t = select(0, data[tid + GROUP_SIZE] - ((sum.x + sum.y + sum.z + sum.w) >> task.data.shift), offs >= ro && offs < bs);
#ifdef BEACCURATE
	t = clamp(t, -0x7fffff, 0x7fffff);
	residual[tid] = (t << 1) ^ (t >> 31);
#else
	res += (t << 1) ^ (t >> 31);
#endif
	barrier(CLK_GLOBAL_MEM_FENCE);

#ifdef BEACCURATE
	if (tid < GROUP_SIZE / 16)
	{
	    __local int4 * chunk = ((__local int4 *)residual) + (tid << 2);
	    int4 sum = chunk[0] + chunk[1] + chunk[2] + chunk[3];
	    int res = sum.x + sum.y + sum.z + sum.w;
	    int k = clamp(27 - clz(res), 0, 14); // 27 - clz(res) == clz(16) - clz(res) == log2(res / 16)
	    len[tid] += (k << 4) + (res >> k);
	}
#endif

	data[tid] = nextData;
    }

#ifdef BEACCURATE
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = GROUP_SIZE / 32; l > 0; l >>= 1)
    {
	if (tid < l)
	    len[tid] += len[tid + l];
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
	output[get_group_id(0)] = len[0] + (bs - ro);
#else
    residual[tid] = res;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = GROUP_SIZE / 2; l > 0; l >>= 1)
    {
	if (tid < l)
	    residual[tid] += residual[tid + l];
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
    {
	int residualLen = (bs - ro);
	float sum = residual[0];// + residualLen / 2;
	//int k = clamp(convert_int_rtn(log2((sum + 0.000001f) / (residualLen + 0.000001f))), 0, 14);
	int k;
	frexp((sum + 0.000001f) / residualLen, &k);
	k = clamp(k - 1, 0, 14);
	output[get_group_id(0)] = residualLen * (k + 1) + convert_int_rtn(min((float)0xffffff, sum / (1 << k)));
    }
#endif
}

__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void cudaChooseBestMethod(
    __global FLACCLSubframeTask *tasks,
    __global int *residual,
    int taskCount
    )
{
    __local struct {
	volatile int index[32];
	volatile int length[32];
    } shared;
    __local FLACCLSubframeData task;
    const int tid = get_local_id(0);
    
    shared.length[tid] = 0x7fffffff;
    shared.index[tid] = tid;
    for (int taskNo = 0; taskNo < taskCount; taskNo++)
    {
	// fetch task data
	if (tid < sizeof(task) / sizeof(int))
	    ((__local int*)&task)[tid] = ((__global int*)(&tasks[taskNo + taskCount * get_group_id(1)].data))[tid];

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tid == 0)
	{
	    // fetch part sum
	    int partLen = residual[taskNo + taskCount * get_group_id(1)];
	    //// calculate part size
	    //int residualLen = task[get_local_id(1)].data.blocksize - task[get_local_id(1)].data.residualOrder;
	    //residualLen = residualLen * (task[get_local_id(1)].data.type != Constant || psum != 0);
	    //// calculate rice parameter
	    //int k = max(0, min(14, convert_int_rtz(log2((psum + 0.000001f) / (residualLen + 0.000001f) + 0.5f))));
	    //// calculate part bit length
	    //int partLen = residualLen * (k + 1) + (psum >> k);

	    int obits = task.obits - task.wbits;
	    shared.length[taskNo] =
		min(obits * task.blocksize,
		    task.type == Fixed ? task.residualOrder * obits + 6 + (4 * 1/2) + partLen :
		    task.type == LPC ? task.residualOrder * obits + 4 + 5 + task.residualOrder * task.cbits + 6 + (4 * 1/2)/* << porder */ + partLen :
		    task.type == Constant ? obits * select(1, task.blocksize, partLen != task.blocksize - task.residualOrder) : 
		    obits * task.blocksize);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
    }
    //shared.index[get_local_id(0)] = get_local_id(0);
    //shared.length[get_local_id(0)] = (get_local_id(0) < taskCount) ? tasks[get_local_id(0) + taskCount * get_group_id(1)].size : 0x7fffffff;

    if (tid < taskCount)
	tasks[tid + taskCount * get_group_id(1)].data.size = shared.length[tid];

    int l1 = shared.length[tid];
    for (int l = 16; l > 0; l >>= 1)
    {
	if (tid < l)
	{
	    int l2 = shared.length[tid + l];
	    shared.index[tid] = shared.index[tid + select(0, l, l2 < l1)];
	    shared.length[tid] = l1 = min(l1, l2);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
	tasks[taskCount * get_group_id(1)].data.best_index = taskCount * get_group_id(1) + shared.index[0];
}

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void cudaCopyBestMethod(
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

__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void cudaCopyBestMethodStereo(
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

// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void cudaEncodeResidual(
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

    __local int4 * cptr = (__local int4 *)&task.coefs[0];
    int4 cptr0 = cptr[0];
#if MAX_ORDER > 4
    int4 cptr1 = cptr[1];
#if MAX_ORDER > 8
    int4 cptr2 = cptr[2];
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
	__local int4 * dptr = (__local int4 *)&data[tid + GROUP_SIZE - ro];
	int4 sum = dptr[0] * cptr0
#if MAX_ORDER > 4
	    + dptr[1] * cptr1
#if MAX_ORDER > 8
	    + dptr[2] * cptr2
#if MAX_ORDER > 12
	    + dptr[3] * cptr[3]
#if MAX_ORDER > 16
	    + dptr[4] * cptr[4]
	    + dptr[5] * cptr[5]
	    + dptr[6] * cptr[6]
	    + dptr[7] * cptr[7]
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

// get_group_id(0) == partition index
// get_group_id(1) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void cudaCalcPartition(
    __global int *partition_lengths,
    __global int *residual,
    __global FLACCLSubframeTask *tasks,
    int max_porder, // <= 8
    int psize // == task.blocksize >> max_porder?
    )
{
    __local int data[GROUP_SIZE];
    __local int length[GROUP_SIZE / 16][16];
    __local FLACCLSubframeData task;

    const int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(1)]))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    int k = tid % 16;
    int x = tid / 16;

    int sum = 0;
    for (int pos0 = 0; pos0 < psize; pos0 += GROUP_SIZE)
    {
	int offs = get_group_id(0) * psize + pos0 + tid;
	// fetch residual
	int s = (offs >= task.residualOrder && pos0 + tid < psize) ? residual[task.residualOffs + offs] : 0;
	// convert to unsigned
	data[tid] = min(0x7fffff, (s << 1) ^ (s >> 31));
	barrier(CLK_LOCAL_MEM_FENCE);

	// calc number of unary bits for each residual sample with each rice paramater
	for (int pos = 0; pos < psize && pos < GROUP_SIZE; pos += GROUP_SIZE / 16)
	    sum += data[pos + x] >> k;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    length[x][k] = min(0x7fffff, sum);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x == 0)
    {
	for (int i = 1; i < GROUP_SIZE / 16; i++)
	    length[0][k] += length[i][k];
	// output length
	const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (k << (max_porder + 1));
	if (k <= 14)
	    partition_lengths[pos + get_group_id(0)] = min(0x7fffff,length[0][k]) + (psize - task.residualOrder * (get_group_id(0) == 0)) * (k + 1);
    }
}

// get_group_id(1) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void cudaCalcPartition16(
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
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(1)]))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;

    if (tid >= ro && tid < 32)
	task.coefs[tid] = 0;

    int k = tid % 16;
    int x = tid / 16;

    barrier(CLK_LOCAL_MEM_FENCE);

    __local int4 * cptr = (__local int4 *)&task.coefs[0];
    int4 cptr0 = cptr[0];
#if MAX_ORDER > 4
    int4 cptr1 = cptr[1];
#if MAX_ORDER > 8
    int4 cptr2 = cptr[2];
#endif
#endif

    data[tid] = 0;
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	int offs = pos + tid;
	// fetch samples
	int nextData = offs < bs ? samples[task.data.samplesOffs + offs] >> task.data.wbits : 0;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// compute residual
	__local int4 * dptr = (__local int4 *)&data[tid + GROUP_SIZE - ro];
	int4 sum = dptr[0] * cptr0
#if MAX_ORDER > 4
	    + dptr[1] * cptr1
#if MAX_ORDER > 8
	    + dptr[2] * cptr2
#if MAX_ORDER > 12
	    + dptr[3] * cptr[3]
#if MAX_ORDER > 16
	    + dptr[4] * cptr[4]
	    + dptr[5] * cptr[5]
	    + dptr[6] * cptr[6]
	    + dptr[7] * cptr[7]
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
	
	s = clamp(s, -0x7fffff, 0x7fffff);
	// convert to unsigned
	res[tid] = (s << 1) ^ (s >> 31);
	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;

	// calc number of unary bits for each residual sample with each rice paramater
	__local int4 * chunk = (__local int4 *)&res[x << 4];
	sum = (chunk[0] >> k) + (chunk[1] >> k) + (chunk[2] >> k) + (chunk[3] >> k);
	s = sum.x + sum.y + sum.z + sum.w;

	const int lpos = (15 << (max_porder + 1)) * get_group_id(1) + (k << (max_porder + 1)) + offs / 16;
	if (k <= 14)
	    partition_lengths[lpos] = min(0x7fffff, s) + (16 - select(0, ro, offs < 16)) * (k + 1);
    }    
}

// Sums partition lengths for a certain k == get_group_id(0)
// Requires 128 threads
// get_group_id(0) == k
// get_group_id(1) == task index
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void cudaSumPartition(
    __global int* partition_lengths,
    int max_porder
    )
{
    __local int data[512]; // max_porder <= 8, data length <= 1 << 9.
    const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (get_group_id(0) << (max_porder + 1));

    // fetch partition lengths
    data[get_local_id(0)] = get_local_id(0) < (1 << max_porder) ? partition_lengths[pos + get_local_id(0)] : 0;
    data[get_local_size(0) + get_local_id(0)] = get_local_size(0) + get_local_id(0) < (1 << max_porder) ? partition_lengths[pos + get_local_size(0) + get_local_id(0)] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    int in_pos = (get_local_id(0) << 1);
    int out_pos = (1 << max_porder) + get_local_id(0);
    for (int bs = 1 << (max_porder - 1); bs > 0; bs >>= 1)
    {
	if (get_local_id(0) < bs) data[out_pos] = data[in_pos] + data[in_pos + 1];
	in_pos += bs << 1;
	out_pos += bs;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) < (1 << max_porder))
	partition_lengths[pos + (1 << max_porder) + get_local_id(0)] = data[(1 << max_porder) + get_local_id(0)];
    if (get_local_size(0) + get_local_id(0) < (1 << max_porder))
	partition_lengths[pos + (1 << max_porder) + get_local_size(0) + get_local_id(0)] = data[(1 << max_porder) + get_local_size(0) + get_local_id(0)];
}

// Finds optimal rice parameter for several partitions at a time.
// get_group_id(0) == chunk index (chunk size is GROUP_SIZE / 8, so total task size is 8 * (2 << max_porder))
// get_group_id(1) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void cudaFindRiceParameter(
    __global int* rice_parameters,
    __global int* partition_lengths,
    int max_porder
    )
{
    __local struct {
	volatile int length[GROUP_SIZE];
	volatile int index[GROUP_SIZE];
    } shared;
    const int tid = get_local_id(0);
    const int ws = GROUP_SIZE / 8;
    const int parts = min(ws, 2 << max_porder);
    const int p = tid % ws;
    const int k = tid / ws; // 0..7
    const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (k << (max_porder + 1));

    // read length for 32 partitions
    int l1 = (p < parts) ? partition_lengths[pos + get_group_id(0) * ws + p] : 0xffffff;
    int l2 = (k + 8 <= 14 && p < parts) ? partition_lengths[pos + (8 << (max_porder + 1)) + get_group_id(0) * ws + p] : 0xffffff;
    // find best rice parameter
    shared.index[tid] = k + ((l2 < l1) << 3);
    shared.length[tid] = l1 = min(l1, l2);
    barrier(CLK_LOCAL_MEM_FENCE);
//#pragma unroll 3
    for (int lsh = GROUP_SIZE / 2; lsh >= ws; lsh >>= 1)
    {
	if (tid < lsh)
	{
	    l2 = shared.length[tid + lsh];
	    shared.index[tid] = shared.index[tid + (l2 < l1) * lsh];
	    shared.length[tid] = l1 = min(l1, l2);
	}    
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid < parts)
    {
	// output rice parameter
	rice_parameters[(get_group_id(1) << (max_porder + 2)) + get_group_id(0) * parts + tid] = shared.index[tid];
	// output length
	rice_parameters[(get_group_id(1) << (max_porder + 2)) + (1 << (max_porder + 1)) + get_group_id(0) * parts + tid] = shared.length[tid];
    }
}

// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void cudaFindPartitionOrder(
    __global int* best_rice_parameters,
    __global FLACCLSubframeTask *tasks,
    __global int* rice_parameters,
    int max_porder
    )
{
    __local struct {
	int length[32];
	int index[32];
    } shared;
    __local int partlen[GROUP_SIZE];
    __local FLACCLSubframeData task;

    const int pos = (get_group_id(0) << (max_porder + 2)) + (2 << max_porder);
    if (get_local_id(0) < sizeof(task) / sizeof(int))
	((__local int*)&task)[get_local_id(0)] = ((__global int*)(&tasks[get_group_id(0)]))[get_local_id(0)];
    // fetch partition lengths
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int porder = max_porder; porder >= 0; porder--)
    {
	int len = 0;
	for (int offs = 0; offs < (1 << porder); offs += GROUP_SIZE)
	    len += offs + get_local_id(0) < (1 << porder) ? rice_parameters[pos + (2 << max_porder) - (2 << porder) + offs + get_local_id(0)] : 0;
	partlen[get_local_id(0)] = len;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int l = min(GROUP_SIZE, 1 << porder) / 2; l > 0; l >>= 1)
	{
	    if (get_local_id(0) < l)
		partlen[get_local_id(0)] += partlen[get_local_id(0) + l];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (get_local_id(0) == 0)
	    shared.length[porder] = partlen[0] + (4 << porder);
	barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (get_local_id(0) < 32 && get_local_id(0) > max_porder)
	shared.length[get_local_id(0)] = 0xfffffff;
    if (get_local_id(0) < 32)
	shared.index[get_local_id(0)] = get_local_id(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    //atom_min(shared.index[get_local_id(0)],);
    int l1 = get_local_id(0) <= max_porder ? shared.length[get_local_id(0)] : 0xfffffff;
    for (int l = 8; l > 0; l >>= 1)
    {
	if (get_local_id(0) < l)
	{
	    int l2 = shared.length[get_local_id(0) + l];
	    shared.index[get_local_id(0)] = shared.index[get_local_id(0) + select(0, l, l2 < l1)];
	    shared.length[get_local_id(0)] = l1 = min(l1, l2);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) == 0)
	tasks[get_group_id(0)].data.porder = shared.index[0];
    if (get_local_id(0) == 0)
    {
	int obits = task.obits - task.wbits;
	tasks[get_group_id(0)].data.size =
	    task.type == Fixed ? task.residualOrder * obits + 6 + l1 :
	    task.type == LPC ? task.residualOrder * obits + 6 + l1 + 4 + 5 + task.residualOrder * task.cbits :
	    task.type == Constant ? obits : obits * task.blocksize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int porder = shared.index[0];
    for (int offs = 0; offs < (1 << porder); offs += GROUP_SIZE)
	if (offs + get_local_id(0) < (1 << porder))
	    best_rice_parameters[(get_group_id(0) << max_porder) + offs + get_local_id(0)] = rice_parameters[pos - (2 << porder) + offs + get_local_id(0)];
    // FIXME: should be bytes?
}
#endif
