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

#if defined(__Cedar__) || defined(__Redwood__) || defined(__Juniper__) || defined(__Cypress__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || defined(__ATI_RV710__)
#define AMD
#endif

#if defined(AMD) && defined(DEBUG)
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#ifdef __CPU__
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

//#if __OPENCL_VERSION__ == 110
#ifdef AMD
#define iclamp(a,b,c) clamp(a,b,c)
#else
#define iclamp(a,b,c) max(b,min(a,c))
#endif

#ifndef M_PI_F
#define M_PI_F M_PI
#endif

#define WARP_SIZE 32

#ifdef HAVE_ATOM
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable
#endif

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
    int headerLen;
    int encodingOffset;
} FLACCLSubframeData;

typedef struct
{
    FLACCLSubframeData data;
    int coefs[32]; // fixme: should be short?
} FLACCLSubframeTask;

#if 0
__kernel void clWindowRectangle(__global float* window, int windowOffset)
{
    window[get_global_id(0)] = 1.0f;
}

__kernel void clWindowFlattop(__global float* window, int windowOffset)
{
    float p = M_PI_F * get_global_id(0) / (get_global_size(0) - 1);
    window[get_global_id(0)] = 1.0f 
	- 1.93f * cos(2 * p)
	+ 1.29f * cos(4 * p) 
	- 0.388f * cos(6 * p) 
	+ 0.0322f * cos(8 * p);
}

__kernel void clWindowTukey(__global float* window, int windowOffset, float p)
{
    int tid = get_global_id(0);
    int Np = (int)(p / 2.0f * get_global_size(0)) - 1;
    int Np2 = tid - (get_global_size(0) - Np - 1) + Np;
    int n = select(max(Np, Np2), tid, tid <= Np);
    window[tid] = 0.5f - 0.5f * cos(M_PI_F * n / Np);
}
#endif

__kernel void clStereoDecorr(
    __global int4 *samples,
    __global int4 *src,
    int offset
)
{
    int pos = get_global_id(0);
    int4 s = src[pos];
    int4 x = (s << 16) >> 16;
    int4 y = s >> 16;
    samples[pos] = x;
    samples[1 * offset + pos] = y;
    samples[2 * offset + pos] = (x + y) >> 1;
    samples[3 * offset + pos] = x - y;
}

__kernel void clChannelDecorr2(
    __global int4 *samples,
    __global int4 *src,
    int offset
)
{
    int pos = get_global_id(0);
    int4 s = src[pos];
    samples[pos] = (s << 16) >> 16;
    samples[offset + pos] = s >> 16;
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

#ifdef __CPU__
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clFindWastedBits(
    __global FLACCLSubframeTask *tasks,
    __global int *samples,
    int tasksPerChannel
)
{
    __global FLACCLSubframeTask* ptask = &tasks[get_group_id(0) * tasksPerChannel];
    int w = 0, a = 0;
    for (int pos = 0; pos < ptask->data.blocksize; pos ++)
    {
	int smp = samples[ptask->data.samplesOffs + pos];
	w |= smp;
	a |= smp ^ (smp >> 31);
    }    
    w = max(0,__ffs(w) - 1);
    a = 32 - clz(a) - w;
    for (int i = 0; i < tasksPerChannel; i++)
    {
	ptask[i].data.wbits = w;
	ptask[i].data.abits = a;
	//ptask[i].data.size = ptask[i].data.obits * ptask[i].data.blocksize;
    }
}
#else
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
    {
	int i = get_group_id(0) * tasksPerChannel + tid;
	tasks[i].data.wbits = w;
	tasks[i].data.abits = a;
	//tasks[i].data.size = tasks[i].data.obits * tasks[i].data.blocksize;
    }
}
#endif

#ifdef __CPU__
#define TEMPBLOCK 128
#define STORE_AC(ro, val) if (ro <= MAX_ORDER) pout[ro] = val;
#define STORE_AC4(ro, val) STORE_AC(ro*4+0, val##ro.x) STORE_AC(ro*4+1, val##ro.y) STORE_AC(ro*4+2, val##ro.z) STORE_AC(ro*4+3, val##ro.w)

// get_num_groups(0) == number of tasks
// get_num_groups(1) == number of windows
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clComputeAutocor(
    __global float *output,
    __global const int *samples,
    __global const float *window,
    __global FLACCLSubframeTask *tasks,
    const int taskCount // tasks per block
)
{
    FLACCLSubframeData task = tasks[get_group_id(0) * taskCount].data;
    int len = task.blocksize;
    int windowOffs = get_group_id(1) * len;
    float data[TEMPBLOCK + MAX_ORDER + 3];
    double4 ac0 = 0.0, ac1 = 0.0, ac2 = 0.0, ac3 = 0.0, ac4 = 0.0, ac5 = 0.0, ac6 = 0.0, ac7 = 0.0, ac8 = 0.0;
    
    for (int pos = 0; pos < len; pos += TEMPBLOCK)
    {
	for (int tid = 0; tid < TEMPBLOCK + MAX_ORDER + 3; tid++)
	    data[tid] = tid < len - pos ? samples[task.samplesOffs + pos + tid] * window[windowOffs + pos + tid] : 0.0f;

	for (int j = 0; j < TEMPBLOCK;)
	{
    	    float4 temp0 = 0.0f, temp1 = 0.0f, temp2 = 0.0f, temp3 = 0.0f, temp4 = 0.0f, temp5 = 0.0f, temp6 = 0.0f, temp7 = 0.0f, temp8 = 0.0f;
	    for (int k = 0; k < 32; k++)
	    {
		float d0 = data[j];
		temp0 += d0 * vload4(0, &data[j]);
		temp1 += d0 * vload4(1, &data[j]);
#if MAX_ORDER >= 8
		temp2 += d0 * vload4(2, &data[j]);
#if MAX_ORDER >= 12
		temp3 += d0 * vload4(3, &data[j]);
#if MAX_ORDER >= 16
		temp4 += d0 * vload4(4, &data[j]);
		temp5 += d0 * vload4(5, &data[j]);
		temp6 += d0 * vload4(6, &data[j]);
		temp7 += d0 * vload4(7, &data[j]);
		temp8 += d0 * vload4(8, &data[j]);
#endif
#endif
#endif
		j++;
	    }
	    ac0 += convert_double4(temp0);
	    ac1 += convert_double4(temp1);
    #if MAX_ORDER >= 8
	    ac2 += convert_double4(temp2);
    #if MAX_ORDER >= 12
	    ac3 += convert_double4(temp3);
    #if MAX_ORDER >= 16
	    ac4 += convert_double4(temp4);
	    ac5 += convert_double4(temp5);
	    ac6 += convert_double4(temp6);
	    ac7 += convert_double4(temp7);
	    ac8 += convert_double4(temp8);
    #endif
    #endif
    #endif
	}
    }
    __global float * pout = &output[(get_group_id(0) * get_num_groups(1) + get_group_id(1)) * (MAX_ORDER + 1)];
    STORE_AC4(0, ac) STORE_AC4(1, ac) STORE_AC4(2, ac) STORE_AC4(3, ac)
    STORE_AC4(4, ac) STORE_AC4(5, ac) STORE_AC4(6, ac) STORE_AC4(7, ac)
    STORE_AC4(8, ac)
}
#else
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

 //   if (tid < GROUP_SIZE / 4)
 //   {
	//float4 dd = 0.0f;
	//if (tid * 4 < bs)
	//    dd = vload4(tid, window + windowOffs) * convert_float4(vload4(tid, samples + task.samplesOffs));
	//vstore4(dd, tid, &data[0]);
 //   }
    data[tid] = 0.0f;
    // This simpler code doesn't work somehow!!!
    //data[tid] = tid < bs ? samples[task.samplesOffs + tid] * window[windowOffs + tid] : 0.0f;

    const int THREADS_FOR_ORDERS = MAX_ORDER < 8 ? 8 : MAX_ORDER < 16 ? 16 : MAX_ORDER < 32 ? 32 : 64;
    float corr = 0.0f;
    float corr1 = 0.0f;
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	// fetch samples
	float nextData = pos + tid < bs ? samples[task.samplesOffs + pos + tid] * window[windowOffs + pos + tid] : 0.0f;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	int lag = tid & (THREADS_FOR_ORDERS - 1);
	int tid1 = tid + GROUP_SIZE - lag;
#ifdef AMD
	float4 res = 0.0f;
	for (int i = 0; i < THREADS_FOR_ORDERS / 4; i++)
	    res += vload4(i, &data[tid1 - lag]) * vload4(i, &data[tid1]);
	corr += res.x + res.y + res.w + res.z;
#else
	float res = 0.0f;
	for (int i = 0; i < THREADS_FOR_ORDERS; i++)
	    res += data[tid1 - lag + i] * data[tid1 + i];
	corr += res;
#endif
	if ((pos & (GROUP_SIZE * 15)) == 0)
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
#endif

#ifdef __CPU__
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clComputeLPC(
    __global float *pautoc,
    __global float *lpcs,
    int windowCount
)
{
    int lpcOffs = (get_group_id(0) + get_group_id(1) * windowCount) * (MAX_ORDER + 1) * 32;
    int autocOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (MAX_ORDER + 1);
    volatile double ldr[32];
    volatile double gen0[32];
    volatile double gen1[32];
    volatile double err[32];
    __global float* autoc = pautoc + autocOffs;
    
    for (int i = 0; i < MAX_ORDER; i++)
    {
	gen0[i] = gen1[i] = autoc[i + 1];
	ldr[i] = 0.0;
    }

    // Compute LPC using Schur and Levinson-Durbin recursion
    double error = autoc[0];
    for (int order = 0; order < MAX_ORDER; order++)
    {
	// Schur recursion
	double reff = -gen1[0] / error;
	//error += gen1[0] * reff; // Equivalent to error *= (1 - reff * reff);
	error *= (1 - reff * reff);
	
	for (int j = 0; j < MAX_ORDER - 1 - order; j++)
	{
	    gen1[j] = gen1[j + 1] + reff * gen0[j];
	    gen0[j] = gen1[j + 1] * reff + gen0[j];
	}

	err[order] = error;

	// Levinson-Durbin recursion

	ldr[order] = reff;
	for (int j = 0; j < order / 2; j++)
	{
	    double tmp = ldr[j];
	    ldr[j] += reff * ldr[order - 1 - j];
	    ldr[order - 1 - j] += reff * tmp;
	}
	if (0 != (order & 1))
	    ldr[order / 2] += ldr[order / 2] * reff;
	
	// Output coeffs
	for (int j = 0; j <= order; j++)
	    lpcs[lpcOffs + order * 32 + j] = -ldr[order - j];
    }
    // Output prediction error estimates
    for (int j = 0; j < MAX_ORDER; j++)
	lpcs[lpcOffs + MAX_ORDER * 32 + j] = err[j];
}
#else
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
#endif

#ifdef __CPU__
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clQuantizeLPC(
    __global FLACCLSubframeTask *tasks,
    __global float*lpcs,
    int taskCount, // tasks per block
    int taskCountLPC, // tasks per set of coeffs (<= 32)
    int minprecision,
    int precisions
)
{
    int bs = tasks[get_group_id(1) * taskCount].data.blocksize;
    int abits = tasks[get_group_id(1) * taskCount].data.abits;
    int lpcOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (MAX_ORDER + 1) * 32;
    float error[MAX_ORDER];
    int best_orders[MAX_ORDER];

    // Load prediction error estimates based on Akaike's Criteria
    for (int tid = 0; tid < MAX_ORDER; tid++)
    {
	error[tid] = bs * log(lpcs[lpcOffs + MAX_ORDER * 32 + tid]) + tid * 4.12f * log(bs);
	best_orders[tid] = tid;
    }
    
    // Select best orders 
    for (int i = 0; i < MAX_ORDER && i < taskCountLPC; i++)
    {
	for (int j = i + 1; j < MAX_ORDER; j++)
	{
	    if (error[best_orders[j]] < error[best_orders[i]])
	    {
		int tmp = best_orders[j];
		best_orders[j] = best_orders[i];
		best_orders[i] = tmp;
	    }
	}				
    }

    // Quantization
    for (int i = 0; i < taskCountLPC; i ++)
    {
	int order = best_orders[i >> precisions];
	int tmpi = 0;
	for (int tid = 0; tid <= order; tid ++)
	{
	    float lpc = lpcs[lpcOffs + order * 32 + tid];
	    // get 15 bits of each coeff
	    int c = convert_int_rte(lpc * (1 << 15));
	    // remove sign bits
	    tmpi |= c ^ (c >> 31);
	}
	// choose precision	
	//int cbits = max(3, min(10, 5 + (abits >> 1))); //  - convert_int_rte(shared.PE[order - 1])
	int cbits = max(3, min(min(13 - minprecision + (i - ((i >> precisions) << precisions)) - (bs <= 2304) - (bs <= 1152) - (bs <= 576), abits), clz(order) + 1 - abits));
	// calculate shift based on precision and number of leading zeroes in coeffs
	int shift = max(0,min(15, clz(tmpi) - 18 + cbits));

	int taskNo = get_group_id(1) * taskCount + get_group_id(0) * taskCountLPC + i;
	tmpi = 0;
	for (int tid = 0; tid <= order; tid ++)
	{
	    float lpc = lpcs[lpcOffs + order * 32 + tid];
	    // quantize coeffs with given shift
	    int c = convert_int_rte(clamp(lpc * (1 << shift), -1 << (cbits - 1), 1 << (cbits - 1)));
	    // remove sign bits
	    tmpi |= c ^ (c >> 31);
	    tasks[taskNo].coefs[tid] = c;
	}
	// calculate actual number of bits (+1 for sign)
	cbits = 1 + 32 - clz(tmpi);
	// output shift, cbits, ro
	tasks[taskNo].data.shift = shift;
	tasks[taskNo].data.cbits = cbits;
	tasks[taskNo].data.residualOrder = order + 1;
    }
}
#else
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
#ifndef HAVE_ATOM
	volatile int tmp[32];
#endif
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
#ifdef HAVE_ATOM
	atom_or(shared.maxcoef + i, coef ^ (coef >> 31));
#else
	shared.tmp[tid] = coef ^ (coef >> 31);
	if (tid < 16)
	{
	    shared.tmp[tid] |= shared.tmp[tid + 16];
	    shared.tmp[tid] |= shared.tmp[tid + 8];
	    shared.tmp[tid] |= shared.tmp[tid + 4];
	    shared.tmp[tid] |= shared.tmp[tid + 2];
	    if (tid == 0)
		shared.maxcoef[i] = shared.tmp[tid] | shared.tmp[tid + 1];
	}
#endif
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
#ifdef HAVE_ATOM
	atom_or(shared.maxcoef2 + i, coef ^ (coef >> 31));
#else
	shared.tmp[tid] = coef ^ (coef >> 31);
	if (tid < 16)
	{
	    shared.tmp[tid] |= shared.tmp[tid + 16];
	    shared.tmp[tid] |= shared.tmp[tid + 8];
	    shared.tmp[tid] |= shared.tmp[tid + 4];
	    shared.tmp[tid] |= shared.tmp[tid + 2];
	    if (tid == 0)
		shared.maxcoef2[i] = shared.tmp[tid] | shared.tmp[tid + 1];
	}
#endif
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
#endif

#ifdef __CPU__
inline int calc_residual(__global int *ptr, int * coefs, int ro)
{
    int sum = 0;
    for (int i = 0; i < ro; i++)
	sum += ptr[i] * coefs[i];
    return sum;
}

#define ENCODE_N(cro,action) for (int pos = cro; pos < bs; pos ++) { \
	int t = (data[pos] - (calc_residual(data + pos - cro, task.coefs, cro) >> task.data.shift)) >> task.data.wbits; \
	action; \
    }
#define SWITCH_N(action) \
    switch (ro) \
    { \
	case 0: ENCODE_N(0, action) break; \
	case 1: ENCODE_N(1, action) break; \
	case 2: ENCODE_N(2, action) /*if (task.coefs[0] == -1 && task.coefs[1] == 2) ENCODE_N(2, 2 * ptr[1] - ptr[0], action) else*/  break; \
	case 3: ENCODE_N(3, action) break; \
	case 4: ENCODE_N(4, action) break; \
	case 5: ENCODE_N(5, action) break; \
	case 6: ENCODE_N(6, action) break; \
	case 7: ENCODE_N(7, action) break; \
	case 8: ENCODE_N(8, action) break; \
	case 9: ENCODE_N(9, action) break; \
	case 10: ENCODE_N(10, action) break; \
	case 11: ENCODE_N(11, action) break; \
	case 12: ENCODE_N(12, action) break; \
	default: ENCODE_N(ro, action) \
    }

__kernel /*__attribute__(( vec_type_hint (int4)))*/ __attribute__((reqd_work_group_size(1, 1, 1)))
void clEstimateResidual(
    __global int*samples,
    __global int*selectedTasks,
    __global FLACCLSubframeTask *tasks
    )
{
    int selectedTask = selectedTasks[get_group_id(0)];
    FLACCLSubframeTask task = tasks[selectedTask];
    int ro = task.data.residualOrder;
    int bs = task.data.blocksize;
#define EPO 6
    int len[1 << EPO]; // blocksize / 64!!!!

    __global int *data = &samples[task.data.samplesOffs];
 //   for (int i = ro; i < 32; i++)
	//task.coefs[i] = 0;
    for (int i = 0; i < 1 << EPO; i++)
	len[i] = 0;

    SWITCH_N((t = clamp(t, -0x7fffff, 0x7fffff), len[pos >> (12 - EPO)] += (t << 1) ^ (t >> 31)))

    int total = 0;
    for (int i = 0; i < 1 << EPO; i++)
    {
	int res = min(0x7fffff,len[i]);
	int k = clamp(clz(1 << (12 - EPO)) - clz(res), 0, 14); // 27 - clz(res) == clz(16) - clz(res) == log2(res / 16)
	total += (k << (12 - EPO)) + (res >> k);
    }
    int partLen = min(0x7ffffff, total) + (bs - ro);
    int obits = task.data.obits - task.data.wbits;
    tasks[selectedTask].data.size = min(obits * bs,
	task.data.type == Fixed ? ro * obits + 6 + (4 * 1/2) + partLen :
	task.data.type == LPC ? ro * obits + 4 + 5 + ro * task.data.cbits + 6 + (4 * 1/2)/* << porder */ + partLen :
	task.data.type == Constant ? obits * select(1, bs, partLen != bs - ro) :
	obits * bs);
}
#else
__kernel /*__attribute__(( vec_type_hint (int4)))*/ __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clEstimateResidual(
    __global int*samples,
    __global int*selectedTasks,
    __global FLACCLSubframeTask *tasks
    )
{
    __local float data[GROUP_SIZE * 2 + 32];
#if !defined(AMD) || !defined(HAVE_ATOM)
    __local volatile int idata[GROUP_SIZE];
#endif
    __local FLACCLSubframeTask task;
    __local int psum[64];
    __local float fcoef[32];
    __local int selectedTask;
    
    if (get_local_id(0) == 0)
	selectedTask = selectedTasks[get_group_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    const int tid = get_local_id(0);
    if (tid < sizeof(task)/sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[selectedTask]))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    int ro = task.data.residualOrder;
    int bs = task.data.blocksize;

    if (tid < 32)
	//fcoef[tid] = select(0.0f, - ((float) task.coefs[tid]) / (1 << task.data.shift), tid < ro);
	fcoef[tid] =  tid < MAX_ORDER && tid + ro - MAX_ORDER >= 0 ? - ((float) task.coefs[tid + ro - MAX_ORDER]) / (1 << task.data.shift) : 0.0f;
    if (tid < 64)
	psum[tid] = 0;
    data[tid] = 0.0f;
    // need to initialize "extra" data, because NaNs can produce wierd results even when multipled by zero extra coefs
    if (tid < 32)
	data[GROUP_SIZE * 2 + tid] = 0.0f;

    int partOrder = max(6, clz(64) - clz(bs - 1) + 1);

    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef AMD
    float4 fc0 = vload4(0, &fcoef[0]);
    float4 fc1 = vload4(1, &fcoef[0]);
#if MAX_ORDER > 8
    float4 fc2 = vload4(2, &fcoef[0]);
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
	__local float* dptr = &data[tid + GROUP_SIZE - MAX_ORDER];
	float4 sum 
#ifdef AMD
	    = fc0 * vload4(0, dptr)
	    + fc1 * vload4(1, dptr)
#else
	    = vload4(0, &fcoef[0]) * vload4(0, dptr)
	    + vload4(1, &fcoef[0]) * vload4(1, dptr)
#endif
#if MAX_ORDER > 8
#ifdef AMD
	    + fc2 * vload4(2, dptr)
#else
	    + vload4(2, &fcoef[0]) * vload4(2, dptr)
#endif
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
	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
	// ensure we're within frame bounds
	t = select(0, t, offs >= ro && offs < bs);
	// overflow protection
	t = iclamp(t, -0x7fffff, 0x7fffff);
	// convert to unsigned
	t = (t << 1) ^ (t >> 31);
#if !defined(AMD) || !defined(HAVE_ATOM)
	// convert to unsigned
	idata[tid] = t;
	barrier(CLK_LOCAL_MEM_FENCE);
	int ps = (1 << partOrder) - 1;
	int lane = tid & ps;
	for (int l = 1 << (partOrder - 1); l > WARP_SIZE; l >>= 1)
	{
            if (lane < l) idata[tid] += idata[tid + l];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lane < WARP_SIZE)
	    for (int l = WARP_SIZE; l > 0; l >>= 1)
		idata[tid] += idata[tid + l];
	if (lane == 0)
	    psum[min(63,offs >> partOrder)] += idata[tid];
#else
	atom_add(&psum[min(63,offs >> partOrder)], t);
#endif
    }

    // calculate rice partition bit length for every (1 << partOrder) samples
    barrier(CLK_LOCAL_MEM_FENCE);
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
	tasks[selectedTask].data.size = len;
    }
}
#endif

__kernel
void clSelectStereoTasks(
    __global FLACCLSubframeTask *tasks,
    __global int*selectedTasks,
    __global int*selectedTasksSecondEstimate,
    __global int*selectedTasksBestMethod,
    int taskCount,
    int selectedCount
    )
{
    int best_size[4];
    for (int ch = 0; ch < 4; ch++)
    {
	int first_no = selectedTasks[(get_global_id(0) * 4 + ch) * selectedCount];
	int best_len = tasks[first_no].data.size;
	for (int i = 1; i < selectedCount; i++)
	{
	    int task_no = selectedTasks[(get_global_id(0) * 4 + ch) * selectedCount + i];
	    int task_len = tasks[task_no].data.size;
	    best_len = min(task_len, best_len);
	}
	best_size[ch] = best_len;
    }
        
    int bitsBest = best_size[2] + best_size[3]; // MidSide
    int chMask = 2 | (3 << 2);
    int bits = best_size[3] + best_size[1];
    chMask = select(chMask, 3 | (1 << 2), bits < bitsBest); // RightSide
    bitsBest = min(bits, bitsBest);
    bits = best_size[0] + best_size[3];
    chMask = select(chMask, 0 | (3 << 2), bits < bitsBest); // LeftSide
    bitsBest = min(bits, bitsBest);
    bits = best_size[0] + best_size[1];
    chMask = select(chMask, 0 | (1 << 2), bits < bitsBest); // LeftRight
    bitsBest = min(bits, bitsBest);
    for (int ich = 0; ich < 2; ich++)
    {
	int ch = select(chMask & 3, chMask >> 2, ich > 0);
	int roffs = tasks[(get_global_id(0) * 4 + ich) * taskCount].data.samplesOffs;
	int nonSelectedNo = 0;
	for (int i = 0; i < taskCount; i++)
	{
	    int no = (get_global_id(0) * 4 + ch) * taskCount + i;
	    selectedTasksBestMethod[(get_global_id(0) * 2 + ich) * taskCount + i] = no;
	    tasks[no].data.residualOffs = roffs;
	    int selectedFound = 0;
	    for(int selectedNo = 0; selectedNo < selectedCount; selectedNo++)
		selectedFound |= (selectedTasks[(get_global_id(0) * 4 + ch) * selectedCount + selectedNo] == no);
	    if (!selectedFound)
		selectedTasksSecondEstimate[(get_global_id(0) * 2 + ich) * (taskCount - selectedCount) + nonSelectedNo++] = no;
	}
    }
}

__kernel
void clChooseBestMethod(
    __global FLACCLSubframeTask *tasks_out,
    __global FLACCLSubframeTask *tasks,
    __global int*selectedTasks,
    int taskCount
    )
{
    int best_no = selectedTasks[get_global_id(0) * taskCount];
    int best_len = tasks[best_no].data.size;
    for (int i = 1; i < taskCount; i++)
    {
	int task_no = selectedTasks[get_global_id(0) * taskCount + i];
	int task_len = tasks[task_no].data.size;
	best_no = select(best_no, task_no, task_len < best_len);
	best_len = min(best_len, task_len);
    }
    tasks_out[get_global_id(0)] = tasks[best_no];
}

#ifdef DO_PARTITIONS
#ifdef __CPU__
// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clEncodeResidual(
    __global int *residual,
    __global int *samples,
    __global FLACCLSubframeTask *tasks
    )
{
    FLACCLSubframeTask task = tasks[get_group_id(0)];
    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;
    __global int *data = &samples[task.data.samplesOffs];
    SWITCH_N(residual[task.data.residualOffs + pos] = t);
}
#else
// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clEncodeResidual(
    __global int *output,
    __global int *samples,
    __global FLACCLSubframeTask *tasks
    )
{
    __local FLACCLSubframeTask task;
    __local int data[GROUP_SIZE * 2 + MAX_ORDER];
    const int tid = get_local_id(0);
    if (get_local_id(0) < sizeof(task) / sizeof(int))
	((__local int*)&task)[get_local_id(0)] = ((__global int*)(&tasks[get_group_id(0)]))[get_local_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;

    if (tid < 32 && tid >= ro)
	task.coefs[tid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef AMD
    int4 cptr0 = vload4(0, &task.coefs[0]);
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
	int4 sum 
#ifdef AMD
            = cptr0 * vload4(0, dptr)
	    + cptr1 * vload4(1, dptr)
#else
            = vload4(0, &task.coefs[0]) * vload4(0, dptr)
	    + vload4(1, &task.coefs[0]) * vload4(1, dptr)
#endif
#if MAX_ORDER > 8
#ifdef AMD
	    + cptr2 * vload4(2, dptr)
#else
	    + vload4(2, &task.coefs[0]) * vload4(2, dptr)
#endif
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
	    ;
	if (off >= ro && off < bs)
	    output[task.data.residualOffs + off] = data[tid + GROUP_SIZE] - ((sum.x + sum.y + sum.z + sum.w) >> task.data.shift);

	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
    }
}
#endif

#ifdef __CPU__
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clCalcPartition(
    __global int *partition_lengths,
    __global int *residual,
    __global FLACCLSubframeTask *tasks,
    int max_porder, // <= 8
    int psize // == task.blocksize >> max_porder?
    )
{
    FLACCLSubframeTask task = tasks[get_group_id(1)];
    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;
    //int psize = bs >> max_porder;
    __global int *pl = partition_lengths + (1 << (max_porder + 1)) * get_group_id(1);

    for (int p = 0; p < (1 << max_porder); p++)
	pl[p] = 0;

    for (int pos = ro; pos < bs; pos ++)
    {
	int t = residual[task.data.residualOffs + pos];
	// overflow protection
	t = clamp(t, -0x7fffff, 0x7fffff);
	// convert to unsigned
	t = (t << 1) ^ (t >> 31);
	pl[pos / psize] += t;
    }
}
#else
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
#endif

#ifdef __CPU__
// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clCalcPartition16(
    __global int *partition_lengths,
    __global int *residual,
    __global int *samples,
    __global FLACCLSubframeTask *tasks,
    int max_porder // <= 8
    )
{
    FLACCLSubframeTask task = tasks[get_global_id(0)];
    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;
    __global int *data = &samples[task.data.samplesOffs];
    __global int *pl = partition_lengths + (1 << (max_porder + 1)) * get_global_id(0);
    for (int p = 0; p < (1 << max_porder); p++)
	pl[p] = 0;
    //__global int *rptr = residual + task.data.residualOffs;
    //SWITCH_N((rptr[pos] = t, pl[pos >> 4] += (t << 1) ^ (t >> 31)));
    SWITCH_N((residual[task.data.residualOffs + pos] = t, t = clamp(t, -0x7fffff, 0x7fffff), t = (t << 1) ^ (t >> 31), pl[pos >> 4] += t));
}
#else
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
	if (k <= 14 && offs < bs)
	    partition_lengths[lpos] = min(0x7fffff, s) + (16 - select(0, ro, offs < 16)) * (k + 1);

//    	if (task.data.blocksize == 16 && x == 0 && k <= 14)
//	    printf("[%d] = %d = s:%d + %d * (k:%d + 1), ro=%d, offs=%d, lpos=%d\n", k, partition_lengths[lpos], s, (16 - select(0, ro, offs < 16)), k, ro, offs, lpos);
    }    
}
#endif

#ifdef __CPU__
// Sums partition lengths for a certain k == get_group_id(0)
// get_group_id(0) == k
// get_group_id(1) == task index
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clSumPartition(
    __global int* partition_lengths,
    int max_porder
    )
{
    if (get_group_id(0) != 0) // ignore k != 0
	return;
    __global int * sums = partition_lengths + (1 << (max_porder + 1)) * get_group_id(1);
    for (int i = max_porder - 1; i >= 0; i--)
    {
	for (int j = 0; j < (1 << i); j++)
	{
	    sums[(2 << i) + j] = sums[2 * j] + sums[2 * j + 1];
	 //   if (get_group_id(1) == 0)
		//printf("[%d][%d]: %d + %d == %d\n", i, j, sums[2 * j], sums[2 * j + 1], sums[2 * j] + sums[2 * j + 1]);
	}
	sums += 2 << i;
    }
}
#else
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
#endif

#ifdef __CPU__
// Finds optimal rice parameter for each partition.
// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clFindRiceParameter(
    __global FLACCLSubframeTask *tasks,
    __global int* rice_parameters,
    __global int* partition_lengths,
    int max_porder
    )
{
    __global FLACCLSubframeTask* task = tasks + get_group_id(0);
    const int tid = get_local_id(0);
    int lim = (2 << max_porder) - 1;
    //int psize = task->data.blocksize >> max_porder;
    int bs = task->data.blocksize;
    int ro = task->data.residualOrder;
    for (int offs = 0; offs < lim; offs ++)
    {
	int pl = partition_lengths[(1 << (max_porder + 1)) * get_group_id(0) + offs];
	int porder = 31 - clz(lim - offs);
	int ps = (bs >> porder) - select(0, ro, offs == lim + 1 - (2 << porder));
	//if (ps <= 0)
	//    printf("max_porder == %d, porder == %d, ro == %d\n", max_porder, porder, ro);
	int k = clamp(31 - clz(pl / max(1, ps)), 0, 14);
	int plk = ps * (k + 1)  + (pl >> k);
	
	// output rice parameter
	rice_parameters[(get_group_id(0) << (max_porder + 2)) + offs] = k;
	// output length
	rice_parameters[(get_group_id(0) << (max_porder + 2)) + (1 << (max_porder + 1)) + offs] = plk;
    }
}
#else
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
#endif

#ifdef __CPU__
// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clFindPartitionOrder(
    __global int *residual,
    __global int* best_rice_parameters,
    __global FLACCLSubframeTask *tasks,
    __global int* rice_parameters,
    int max_porder
    )
{
    __global FLACCLSubframeTask* task = tasks + get_group_id(0);
    int partlen[9];
    for (int p = 0; p < 9; p++)
	partlen[p] = 0;
    // fetch partition lengths
    const int pos = (get_group_id(0) << (max_porder + 2)) + (2 << max_porder);
    int lim = (2 << max_porder) - 1;
    for (int offs = 0; offs < lim; offs ++)
    {
	int len = rice_parameters[pos + offs];
	int porder = 31 - clz(lim - offs);
	partlen[porder] += len;
    }

    int best_length = partlen[0] + 4;
    int best_porder = 0;
    for (int porder = 1; porder <= max_porder; porder++)
    {
	int length = (4 << porder) + partlen[porder];
	best_porder = select(best_porder, porder, length < best_length);
	best_length = min(best_length, length);
    }

    best_length = (4 << best_porder) + task->data.blocksize - task->data.residualOrder;
    int best_psize = task->data.blocksize >> best_porder;
    int start = task->data.residualOffs + task->data.residualOrder;
    int fin = task->data.residualOffs + best_psize;
    for (int p = 0; p < (1 << best_porder); p++)
    {
	int k = rice_parameters[pos - (2 << best_porder) + p];
	best_length += k * (fin - start);
	for (int i = start; i < fin; i++)
	{
	    int t = residual[i];
	    best_length += ((t << 1) ^ (t >> 31)) >> k;
	}
	start = fin;
	fin += best_psize;
    }

    int obits = task->data.obits - task->data.wbits;
    task->data.porder = best_porder;
    task->data.headerLen = 
	task->data.type == Constant ? obits :
	task->data.type == Verbatim ? obits * task->data.blocksize :
	task->data.type == Fixed ? task->data.residualOrder * obits + 6 :
	task->data.type == LPC ? task->data.residualOrder * obits + 6 + 4 + 5 + task->data.residualOrder * task->data.cbits : 0;
    task->data.size = 
	task->data.headerLen + ((task->data.type == Fixed || task->data.type == LPC) ? best_length : 0);
    if (task->data.size >= obits * task->data.blocksize)
    {
	task->data.headerLen = task->data.size = obits * task->data.blocksize;
	task->data.type = Verbatim;
    }
    for (int offs = 0; offs < (1 << best_porder); offs ++)
	best_rice_parameters[(get_group_id(0) << max_porder) + offs] = rice_parameters[pos - (2 << best_porder) + offs];
}
#else
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
	task.porder = best_porder;
	int obits = task.obits - task.wbits;
	task.headerLen = 
	    task.type == Fixed ? task.residualOrder * obits + 6 :
	    task.type == LPC ? task.residualOrder * obits + 6 + 4 + 5 + task.residualOrder * task.cbits :
	    task.type == Constant ? obits :
	    /* task.type == Verbatim ? */ obits * task.blocksize;
	task.size = task.headerLen + select(0, best_length, task.type == Fixed || task.type == LPC);
	if (task.size >= obits * task.blocksize)
	{
	    task.headerLen = task.size = obits * task.blocksize;
	    task.type = Verbatim;
	}
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < sizeof(task) / sizeof(int))
	((__global int*)(&tasks[get_group_id(0)]))[get_local_id(0)] = ((__local int*)&task)[get_local_id(0)];
    for (int offs = get_local_id(0); offs < (1 << best_porder); offs += GROUP_SIZE)
	best_rice_parameters[(get_group_id(0) << max_porder) + offs] = rice_parameters[pos - (2 << best_porder) + offs];
    // FIXME: should be bytes?
}
#endif

#ifdef DO_RICE
#ifdef __CPU__
typedef struct BitWriter_t
{
    __global int *buffer;
    unsigned int bit_buf;
    int bit_left;
    int buf_ptr;
} BitWriter;

inline void writebits(BitWriter *bw, int bits, int v)
{
    uint val = ((uint)v) & ((1 << bits) - 1);
    if (bits < bw->bit_left)
    {
	bw->bit_buf = (bw->bit_buf << bits) | val;
	bw->bit_left -= bits;
    }
    else
    {
//	if (bits >= 32) printf("\n\n\n\n-------------------------\n\n\n");
	unsigned int bb = (bw->bit_buf << bw->bit_left) | (val >> (bits - bw->bit_left));
	bw->buffer[bw->buf_ptr++] = (bb >> 24) | ((bb >> 8) & 0xff00) | ((bb << 8) & 0xff0000) | ((bb << 24) & 0xff000000);
	bw->bit_left += (32 - bits);
	bw->bit_buf = val;
//	bw->bit_buf = val & ((1 << (32 - bw->bit_left)) - 1);
    }
}

inline void flush(BitWriter *bw)
{
    if (bw->bit_left < 32)
	writebits(bw, bw->bit_left, 0);
}
#endif

inline int len_utf8(int n)
{
    int bts = 31 - clz(n);
    if (bts < 7)
	return 8;
    return 8 * ((bts + 4) / 5);
}

// get_global_id(0) * channels == task index
__kernel 
void clCalcOutputOffsets(
    __global int *residual,
    __global int *samples,
    __global FLACCLSubframeTask *tasks,
    int channels,
    int frameCount,
    int firstFrame
    )
{
    int offset = 0;
    for (int iFrame = 0; iFrame < frameCount; iFrame++)
    {
	//printf("len_utf8(%d) == %d\n", firstFrame + iFrame, len_utf8(firstFrame + iFrame));
	offset += 15 + 1 + 4 + 4 + 4 + 3 + 1 + len_utf8(firstFrame + iFrame)
	    // + 8-16 // custom block size
	    // + 8-16 // custom sample rate
	    ;
	int bs = tasks[iFrame * channels].data.blocksize;
	//public static readonly int[] flac_blocksizes = new int[15] { 0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384 };
	if (bs != 4096 && bs != 4608) // TODO: check all other standard sizes
	    offset += select(8, 16, bs >= 256);
	
	// assert (offset % 8) == 0
	offset += 8;
	for (int ch = 0; ch < channels; ch++)
	{
	    __global FLACCLSubframeTask* task = tasks + iFrame * channels + ch;
	    offset += 8 + task->data.wbits;
	    task->data.encodingOffset = offset + task->data.headerLen;
	    offset += task->data.size;
	}
	offset = (offset + 7) & ~7;
	offset += 16;
    }
}

// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clRiceEncoding(
    __global int *residual,
    __global int *samples,
    __global int* best_rice_parameters,
    __global FLACCLSubframeTask *tasks,
    __global unsigned int* output,
    int max_porder
    )
{
#ifdef __CPU__
    __global FLACCLSubframeTask* task = tasks + get_group_id(0);
    if (task->data.type == Fixed || task->data.type == LPC)
    {
	int ro = task->data.residualOrder;
	int bs = task->data.blocksize;
	int porder = task->data.porder;
	int psize = bs >> porder;
	
	BitWriter bw;
	bw.buffer = output;
	bw.buf_ptr = task->data.encodingOffset / 32;
	bw.bit_left = 32 - (task->data.encodingOffset & 31);
	bw.bit_buf = 0;

	//if (get_group_id(0) == 0) printf("%d\n", offs);

	int res_cnt = psize - ro;
	// residual
	int j = ro;
	__global int * kptr = &best_rice_parameters[get_group_id(0) << max_porder];
	for (int p = 0; p < (1 << porder); p++)
	{
	    int k = kptr[p];
	    writebits(&bw, 4, k);
	    //if (get_group_id(0) == 0) printf("[%x] ", k);
	    //if (get_group_id(0) == 0) printf("(%x) ", bw.bit_buf);
	    if (p == 1) res_cnt = psize;
	    int cnt = min(res_cnt, bs - j);
	    for (int i = 0; i < cnt; i++)
	    {
		int v = residual[task->data.residualOffs + j + i];
		v = (v << 1) ^ (v >> 31);
		// write quotient in unary
		int q = (v >> k) + 1;
		int bits = k + q;
		while (bits > 31)
		{
		    int b = min(bits - 31, 31);
		    if (b < bw.bit_left)
		    {
			bw.bit_buf <<= b;
			bw.bit_left -= b;
		    }
		    else
		    {
			unsigned int bb = bw.bit_buf << bw.bit_left;
			bw.bit_buf = 0;
			bw.bit_left += (32 - b);
			bw.buffer[bw.buf_ptr++] = (bb >> 24) | ((bb >> 8) & 0xff00) | ((bb << 8) & 0xff0000) | ((bb << 24) & 0xff000000);
		    }
		    bits -= b;
		}
		unsigned int val = (unsigned int)((v & ((1 << k) - 1)) | (1 << k));
		if (bits < bw.bit_left)
		{
		    bw.bit_buf = (bw.bit_buf << bits) | val;
		    bw.bit_left -= bits;
		}
		else
		{
		    unsigned int bb = (bw.bit_buf << bw.bit_left) | (val >> (bits - bw.bit_left));
		    bw.bit_buf = val;
		    bw.bit_left += (32 - bits);
		    bw.buffer[bw.buf_ptr++] = (bb >> 24) | ((bb >> 8) & 0xff00) | ((bb << 8) & 0xff0000) | ((bb << 24) & 0xff000000);
		}
		////if (get_group_id(0) == 0) printf("%x ", v);
		//writebits(&bw, (v >> k) + 1, 1);
		////if (get_group_id(0) == 0) printf("(%x) ", bw.bit_buf);
		//writebits(&bw, k, v);
		////if (get_group_id(0) == 0) printf("(%x) ", bw.bit_buf);
	    }
	    j += cnt;
	}
	//if (bw.buf_ptr * 32 + 32 - bw.bit_left != task->data.encodingOffset - task->data.headerLen + task->data.size) 
	//    printf("bit length mismatch: encodingOffset == %d, headerLen == %d, size == %d, so should be %d, but is %d\n",
	//	task->data.encodingOffset, task->data.headerLen, task->data.size,
	//	task->data.encodingOffset - task->data.headerLen + task->data.size,
	//	bw.buf_ptr * 32 + 32 - bw.bit_left
	//	);
	//if (get_group_id(0) == 0) printf("\n");
	flush(&bw);
    }
#else
    __local unsigned int data[GROUP_SIZE];
    __local int mypos[GROUP_SIZE+1];
    __local FLACCLSubframeData task;

    int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0)]))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0)
	mypos[GROUP_SIZE] = 0;
    data[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    const int bs = task.blocksize;
    int start = task.encodingOffset;
    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	int offs = pos + tid;
	int v = offs < bs ? residual[task.residualOffs + offs] : 0;
	int part = (offs << task.porder) / bs;
	int k = offs < bs ? best_rice_parameters[(get_group_id(0) << max_porder) + part] : 0;
	int pstart = offs == task.residualOrder || offs == ((part * bs) >> task.porder);
	v = (v << 1) ^ (v >> 31);
	int mylen = select(0, (v >> k) + 1 + k + select(0, 4, pstart), offs >= task.residualOrder && offs < bs);
	mypos[tid] = mylen;
	// Inclusive scan(+)
#if 0
	int lane = (tid & (WARP_SIZE - 1));
	for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
	    mypos[tid] += mypos[select(GROUP_SIZE, tid - offset, lane >= offset)];
#if 1
	barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = GROUP_SIZE - WARP_SIZE; j > 0; j -= WARP_SIZE)
	{
	    if (tid >= j)
		mypos[tid] += mypos[j - 1];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
#else
	if ((tid & (WARP_SIZE - 1)) == WARP_SIZE - 1)
	    warppos[tid/WARP_SIZE] = mypos[tid];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int offset = 1; offset < GROUP_SIZE/WARP_SIZE; offset <<= 1)
	{
	    if (offset <= tid && tid < GROUP_SIZE/WARP_SIZE)
		warppos[tid] += warppos[tid - offset];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	mypos[tid] += tid / WARP_SIZE == 0 ? 0 : warppos[tid / WARP_SIZE - 1];
#endif
#else
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int offset = 1; offset < GROUP_SIZE; offset <<= 1)
	{
	    int t = tid >= offset ? mypos[tid - offset] : 0;
	    barrier(CLK_LOCAL_MEM_FENCE);
	    mypos[tid] += t;
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
#endif
	mypos[tid] += start;
	int start32 = start / 32;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (pstart && mylen)
	{
	    int kpos = mypos[tid] - mylen;
	    int kpos0 = (kpos >> 5) - start32;
	    int kpos1 = kpos & 31;
	    unsigned int kval = (unsigned int)k << 28;
	    unsigned int kval0 = kval >> kpos1;
	    unsigned int kval1 = select(0U, kval << (32 - kpos1), kpos1);
	    atom_or(&data[kpos0], kval0);
	    atom_or(&data[kpos0 + 1], kval1);
	}
	int qpos = mypos[tid] - k - 1;
	int qpos0 = (qpos >> 5) - start32;
	int qpos1 = qpos & 31;
	unsigned int qval = select(0U, (1U << 31) | ((unsigned int)v << (31 - k)), mylen);
	unsigned int qval0 = qval >> qpos1;
	unsigned int qval1= select(0U, qval << (32 - qpos1), qpos1);
	atom_or(&data[qpos0], qval0);
	atom_or(&data[qpos0 + 1], qval1);
	start = mypos[GROUP_SIZE - 1];
	barrier(CLK_LOCAL_MEM_FENCE);
	unsigned int bb = data[tid];
//	bb = (bb >> 24) | ((bb >> 8) & 0xff00U) | ((bb << 8) & 0xff0000U) | (bb << 24);
	if ((start32 + tid) * 32 <= start)
	    output[start32 + tid] = 0U;
	unsigned int remainder = data[start / 32 - start32];
	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = select(0U, remainder, tid == 0);
    }
 //   if (tid == 0 && start != task.encodingOffset - task.headerLen + task.size)
	//printf("size mismatch: %d != %d\n", start, task.encodingOffset - task.headerLen + task.size);
#endif
}
#endif /* DO_RICE */
#endif /* DO_PARTITIONS */
#endif
