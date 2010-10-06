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
    for (int pos = 0; pos < task.blocksize; pos += get_local_size(0))
    {
	int smp = pos + tid < task.blocksize ? samples[task.samplesOffs + pos + tid] : 0;
	w |= smp;
	a |= smp ^ (smp >> 31);
    }
    wbits[tid] = w;
    abits[tid] = a;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1)
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
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int order = 0; order < MAX_ORDER; order++)
    {
	// Schur recursion
	float reff = -shared.gen1[0] / error;
	error += shared.gen1[0] * reff; // Equivalent to error *= (1 - reff * reff);
	float gen1;
	if (get_local_id(0) < MAX_ORDER - 1 - order)
	{
	    gen1 = shared.gen1[get_local_id(0) + 1] + reff * gen0;
	    gen0 += shared.gen1[get_local_id(0) + 1] * reff;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_local_id(0) < MAX_ORDER - 1 - order)
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
    shared.index[32 + tid] = min(MAX_ORDER - 1, tid);
    shared.error[32 + tid] = shared.task.blocksize * 64 + tid;

    // Load prediction error estimates
    if (tid < MAX_ORDER)
	shared.error[tid] = shared.task.blocksize * log(lpcs[shared.lpcOffs + MAX_ORDER * 32 + tid]) + tid * 5.12f * log(shared.task.blocksize);
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
	if (i < taskCountLPC)
	{
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
}

__kernel /*__attribute__(( vec_type_hint (int4)))*/ __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void cudaEstimateResidual(
    __global int*output,
    __global int*samples,
    __global FLACCLSubframeTask *tasks
    )
{
    __local float data[GROUP_SIZE * 2];
    __local int residual[GROUP_SIZE];
    __local FLACCLSubframeTask task;
    __local float4 coefsf4[8];

    const int tid = get_local_id(0);
    if (tid < sizeof(task)/sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0)]))[tid];
    barrier(CLK_GLOBAL_MEM_FENCE);

    int ro = task.data.residualOrder;
    int bs = task.data.blocksize;
    float res = 0;

    if (tid < 32)
	((__local float *)&coefsf4[0])[tid] = select(0.0f, ((float)task.coefs[tid]) / (1 << task.data.shift), tid < ro);
    data[tid] = tid < bs ? (float)(samples[task.data.samplesOffs + tid] >> task.data.wbits) : 0.0f;
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	// fetch samples
	float nextData = pos + tid + GROUP_SIZE < bs ? (float)(samples[task.data.samplesOffs + pos + tid + GROUP_SIZE] >> task.data.wbits) : 0.0f;
	data[tid + GROUP_SIZE] = nextData;
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

    int residualLen = (bs - ro) / GROUP_SIZE + select(0, 1, tid < (bs - ro) % GROUP_SIZE);
    int k = clamp(convert_int_rtn(log2((res + 0.000001f) / (residualLen + 0.000001f))), 0, 14);
    residual[tid] = residualLen * (k + 1) + (convert_int_rtz(res) >> k);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = GROUP_SIZE / 2; l > 0; l >>= 1)
    {
	if (tid < l)
	    residual[tid] += residual[tid + l];
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
	output[get_group_id(0)] = residual[0];
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
		    task.type == Constant ? obits * (1 + task.blocksize * (partLen != 0)) : 
		    obits * task.blocksize);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
    }
    //shared.index[get_local_id(0)] = get_local_id(0);
    //shared.length[get_local_id(0)] = (get_local_id(0) < taskCount) ? tasks[get_local_id(0) + taskCount * get_group_id(1)].size : 0x7fffffff;

    if (tid < taskCount)
	tasks[tid + taskCount * get_group_id(1)].data.size = shared.length[tid];

    int l1 = shared.length[tid];
    for (int sh = 4; sh > 0; sh --)
    {
	if (tid < (1 << sh))
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
	((__local int*)&task)[get_local_id(0)] = ((__global int*)(&tasks[get_group_id(1)]))[get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;

    data[tid] = tid < bs ? samples[task.data.samplesOffs + tid] >> task.data.wbits : 0;
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	// fetch samples
	float nextData = pos + tid + GROUP_SIZE < bs ? samples[task.data.samplesOffs + pos + tid + GROUP_SIZE] >> task.data.wbits : 0;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	// compute residual
	int sum = 0;
	for (int c = 0; c < ro; c++)
	    sum += data[tid + c] * task.coefs[c];
	sum = data[tid + ro] - (sum >> task.data.shift);
	if (pos + tid + ro < bs)
	    output[task.data.residualOffs + pos + tid + ro] = sum;

	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
    }
}

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

    int k = tid % (GROUP_SIZE / 16);
    int x = tid / (GROUP_SIZE / 16);

    int sum = 0;
    for (int pos0 = 0; pos0 < psize; pos0 += GROUP_SIZE)
    {
	int offs = get_group_id(0) * psize + pos0 + tid;
	// fetch residual
	int s = (offs >= task.residualOrder && pos0 + tid < psize) ? residual[task.residualOffs + offs] : 0;
	// convert to unsigned
	data[tid] = min(0xfffff, (s << 1) ^ (s >> 31));
	barrier(CLK_LOCAL_MEM_FENCE);

	// calc number of unary bits for each residual sample with each rice paramater
	for (int pos = 0; pos < psize && pos < GROUP_SIZE; pos += GROUP_SIZE / 16)
	    sum += data[pos + x] >> k;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    length[x][k] = min(0xfffff, sum);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x == 0)
    {
	for (int i = 1; i < GROUP_SIZE / 16; i++)
	    length[0][k] += length[i][k];
	// output length
	const int pos = (15 << (max_porder + 1)) * get_group_id(1) + (k << (max_porder + 1));
	if (k <= 14)
	    partition_lengths[pos + get_group_id(0)] = min(0xfffff,length[0][k]) + (psize - task.residualOrder * (get_group_id(0) == 0)) * (k + 1);
    }
}

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
