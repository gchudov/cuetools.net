/**
 * CUETools.FLACCL: FLAC audio encoder using OpenCL
 * Copyright (c) 2010-2023 Gregory S. Chudov
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

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || defined(__Capeverde__)
#define AMD
#elif defined(__Cayman__) || defined(__Barts__) || defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || defined(__Cedar__)
#define AMD
#elif defined(__ATI_RV770__) || defined(__ATI_RV730__) || defined(__ATI_RV710__)
#define AMD
#endif

#define VENDOR_ID_INTEL  0x8086
#define VENDOR_ID_NVIDIA 0x10DE
#define VENDOR_ID_ATIAMD 0x1002

#ifndef FLACCL_CPU
#if VENDOR_ID == VENDOR_ID_INTEL
#define WARP_SIZE 16
#else
#define WARP_SIZE 32
#endif
#endif

#if defined(HAVE_cl_khr_fp64) || defined(HAVE_cl_amd_fp64)
#define HAVE_DOUBLE
#define ZEROD 0.0
//#define FAST_DOUBLE
#else
#define double float
#define double4 float4
#define ZEROD 0.0f
#endif
#if defined(HAVE_DOUBLE) && defined(FAST_DOUBLE)
#define fastdouble double
#define fastdouble4 double4
#define ZEROFD 0.0
#else
#define fastdouble float
#define fastdouble4 float4
#define ZEROFD 0.0f
#endif

#if BITS_PER_SAMPLE > 16
#define MAX_RICE_PARAM 30
#define RICE_PARAM_BITS 5
#else
#define MAX_RICE_PARAM 14
#define RICE_PARAM_BITS 4
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
    int coding_method;
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

#if BITS_PER_SAMPLE > 16
__kernel void clStereoDecorr(
    __global int *samples,
    __global unsigned char *src,
    int offset
)
{
    int pos = get_global_id(0);
    int bpos = pos * 6;
    int x = (((int)src[bpos] << 8) | ((int)src[bpos+1] << 16) | ((int)src[bpos+2] << 24)) >> 8;
    int y = (((int)src[bpos+3] << 8) | ((int)src[bpos+4] << 16) | ((int)src[bpos+5] << 24)) >> 8;
    samples[pos] = x;
    samples[1 * offset + pos] = y;
    samples[2 * offset + pos] = (x + y) >> 1;
    samples[3 * offset + pos] = x - y;
}

__kernel void clChannelDecorr2(
    __global int *samples,
    __global unsigned char *src,
    int offset
)
{
    int pos = get_global_id(0);
    int bpos = pos * 6;
    samples[pos] = (((int)src[bpos] << 8) | ((int)src[bpos+1] << 16) | ((int)src[bpos+2] << 24)) >> 8;
    samples[offset + pos] = (((int)src[bpos+3] << 8) | ((int)src[bpos+4] << 16) | ((int)src[bpos+5] << 24)) >> 8;
}

__kernel void clChannelDecorrX(
    __global int *samples,
    __global unsigned char *src,
    int offset
)
{
    int pos = get_global_id(0);
    for (int ch = 0; ch < MAX_CHANNELS; ch++)
    {
	int bpos = 3 * (pos * MAX_CHANNELS + ch);
	samples[offset * ch + pos] = (((int)src[bpos] << 8) | ((int)src[bpos+1] << 16) | ((int)src[bpos+2] << 24)) >> 8;
    }
}
#else
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

__kernel void clChannelDecorrX(
    __global int *samples,
    __global short *src,
    int offset
)
{
    int pos = get_global_id(0);
    for (int ch = 0; ch < MAX_CHANNELS; ch++)
    {
	int bpos = pos * MAX_CHANNELS + ch;
	samples[offset * ch + pos] = src[bpos];
    }
}
#endif

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

#ifdef FLACCL_CPU
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

#ifdef FLACCL_CPU
#define TEMPBLOCK 512
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
    __local fastdouble data[GROUP_SIZE * 2];
    __local FLACCLSubframeData task;
    const int tid = get_local_id(0);
    // fetch task data
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(tasks + taskCount * get_group_id(0)))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int bs = task.blocksize;
    data[tid] = ZEROFD;

    const int THREADS_FOR_ORDERS = MAX_ORDER < 8 ? 8 : MAX_ORDER < 16 ? 16 : MAX_ORDER < 32 ? 32 : 64;
    int lag = tid & (THREADS_FOR_ORDERS - 1);
    int tid1 = tid + GROUP_SIZE - lag;
    int pos = 0;
    const __global float * wptr = &window[get_group_id(1) * bs];
//    const __global int * sptr = &samples[task.samplesOffs];
    double corr = ZEROD;

    for (pos = 0; pos + GROUP_SIZE - 1 < bs; pos += GROUP_SIZE)
    {
	int off = pos + tid;
	// fetch samples
	fastdouble nextData = samples[task.samplesOffs + off] * wptr[off];
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	fastdouble4 tmp = ZEROFD;
	for (int i = 0; i < THREADS_FOR_ORDERS / 4; i++)
	    tmp += vload4(i, &data[tid1 - lag]) * vload4(i, &data[tid1]);
	corr += (tmp.x + tmp.y) + (tmp.w + tmp.z);

	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
    }
    {
	int off = pos + tid;
	// fetch samples
	double nextData = off < bs ? samples[task.samplesOffs + off] * wptr[off] : ZEROD;
	data[tid + GROUP_SIZE] = nextData;
        barrier(CLK_LOCAL_MEM_FENCE);

	fastdouble4 tmp = ZEROFD;
	for (int i = 0; i < THREADS_FOR_ORDERS / 4; i++)
	    tmp += vload4(i, &data[tid1 - lag]) * vload4(i, &data[tid1]);
	corr += (tmp.x + tmp.y) + (tmp.w + tmp.z);
    }

    data[tid] = corr;
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

#ifdef FLACCL_CPU
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
	volatile double ldr[32];
	volatile double gen1[32];
	volatile float error[32];
	volatile float autoc[33];
    } shared;
    const int tid = get_local_id(0);// + get_local_id(1) * 32;    
    int autocOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (MAX_ORDER + 1);
    int lpcOffs = autocOffs * 32;
    
    shared.autoc[get_local_id(0)] = get_local_id(0) <= MAX_ORDER ? autoc[autocOffs + get_local_id(0)] : 0;
    if (get_local_id(0) + get_local_size(0) <= MAX_ORDER)
	shared.autoc[get_local_id(0) + get_local_size(0)] = autoc[autocOffs + get_local_id(0) + get_local_size(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute LPC using Schur and Levinson-Durbin recursion
    double gen0 = shared.gen1[get_local_id(0)] = shared.autoc[get_local_id(0)+1];
    shared.ldr[get_local_id(0)] = ZEROD;
    double error = shared.autoc[0];
    
#ifdef DEBUGPRINT1
    int magic = autocOffs == 0; // shared.autoc[0] == 177286873088.0f;
    if (magic && get_local_id(0) <= MAX_ORDER)
        printf("autoc[%d] == %f\n", get_local_id(0), shared.autoc[get_local_id(0)]);
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int order = 0; order < MAX_ORDER; order++)
    {
	// Schur recursion
	double reff = -shared.gen1[0] / error;
	//error += shared.gen1[0] * reff; // Equivalent to error *= (1 - reff * reff);
	error *= (1 - reff * reff);
	double gen1;
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
	double ldr = shared.ldr[get_local_id(0)];
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
#ifdef DEBUGPRINT1
    if (magic && get_local_id(0) < MAX_ORDER)
        printf("error[%d] == %f\n", get_local_id(0), shared.error[get_local_id(0)]);
#endif
    // Output prediction error estimates
    if (get_local_id(0) < MAX_ORDER)
	lpcs[lpcOffs + MAX_ORDER * 32 + get_local_id(0)] = shared.error[get_local_id(0)];
}
#endif

#ifdef FLACCL_CPU
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
    int obits = tasks[get_group_id(1) * taskCount].data.obits;
    int lpcOffs = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * (MAX_ORDER + 1) * 32;
    float error[MAX_ORDER];
    int best_orders[MAX_ORDER];

    int best8 = 0;
    // Load prediction error estimates based on Akaike's Criteria
    for (int tid = 0; tid < MAX_ORDER; tid++)
    {
	error[tid] = bs * log(1.0f + lpcs[lpcOffs + MAX_ORDER * 32 + tid]) + tid * 4.12f * log((float)bs);
	best_orders[tid] = tid;
	if (tid < 8 && error[tid] < error[best8])
	    best8 = tid;
    }

#if 0
    for (int i = best8 + 1; i < MAX_ORDER; i++)
	error[i] += 20.5f * log((float)bs);
#endif
    
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
#if BITS_PER_SAMPLE > 16
	int cbits = max(3, min(15 - minprecision + (i - ((i >> precisions) << precisions)) - (bs <= 2304) - (bs <= 1152) - (bs <= 576), abits));
#else
	int cbits = max(3, min(min(13 - minprecision + (i - ((i >> precisions) << precisions)) - (bs <= 2304) - (bs <= 1152) - (bs <= 576), abits), clz(order) + 1 - obits));
#endif
	// calculate shift based on precision and number of leading zeroes in coeffs
	int shift = max(0,min(15, clz(tmpi) - 18 + cbits));

	int taskNo = get_group_id(1) * taskCount + get_group_id(0) * taskCountLPC + i;
	tmpi = 0;
	for (int tid = 0; tid <= order; tid ++)
	{
	    float lpc = lpcs[lpcOffs + order * 32 + tid];
	    // quantize coeffs with given shift
		int c = convert_int_rte(clamp(lpc * (1 << shift), (float)((-1 << (cbits - 1)) + 1), (float)((1 << (cbits - 1)) - 1)));
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
//    	volatile int best8;
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

    // Load prediction error estimates
    if (tid < MAX_ORDER)
	shared.error[tid] = shared.task.blocksize * log(lpcs[lpcOffs + MAX_ORDER * 32 + tid]) + tid * 4.12f * log((float)shared.task.blocksize);
	//shared.error[get_local_id(0)] = shared.task.blocksize * log(lpcs[lpcOffs + MAX_ORDER * 32 + get_local_id(0)]) + get_local_id(0) * 0.30f * (shared.task.abits + 1) * log(shared.task.blocksize);
#if 0
    if (tid == 0)
    {
	int b8 = 0;
        for (int i = 1; i < 8; i++)
	    if (shared.error[i] < shared.error[b8])
		b8 = i;
	shared.best8 = b8;
    }
    shared.error[tid] += select(0.0f, 20.5f * log((float)shared.task.blocksize), tid > shared.best8);
#endif
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
	atomic_or(shared.maxcoef + i, coef ^ (coef >> 31));
	barrier(CLK_LOCAL_MEM_FENCE);

        int cbits = min(51 - 2 * clz(shared.task.blocksize), shared.task.abits) - minprecision + (i - ((i >> precisions) << precisions));
#if BITS_PER_SAMPLE <= 16
	// Limit cbits so that 32-bit arithmetics will be enough when calculating residual
        // (1 << (obits - 1)) * ((1 << (cbits - 1)) - 1) * (order + 1) < (1 << 31)
        // (1 << (cbits - 1)) - 1 < (1 << (32 - obits)) / (order + 1)
        // (1 << (cbits - 1)) <= (1 << (32 - obits)) / (order + 1)
        // (1 << (cbits - 1)) <= (1 << (32 - obits - (32 - clz(order))) <= (1 << (32 - obits)) / (order + 1)
        // (1 << (cbits - 1)) <= (1 << (clz(order) - obits))
        // cbits - 1 <= clz(order) - obits
        // cbits <= clz(order) - obits + 1
        cbits = min(cbits, clz(order) + 1 - shared.task.obits);
#endif
        cbits = clamp(cbits, 3, 15);

	// Calculate shift based on precision and number of leading zeroes in coeffs.
	// We know that if shifted by 15, coefs require 
	// 33 - clz(shared.maxcoef[i]) bits;
	// So to get the desired cbits, we need to shift coefs by 
	// 15 + cbits - (33 - clz(shared.maxcoef[i]));
	int shift = clamp(clz(shared.maxcoef[i]) - 18 + cbits, 0, 15);

	int lim = (1 << (cbits - 1)) - 1;
	coef = clamp(convert_int_rte(lpc * (1 << shift)), -lim, lim);

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

#ifdef FLACCL_CPU
#define TEMPBLOCK1 TEMPBLOCK

__kernel __attribute__(( vec_type_hint (int4))) __attribute__((reqd_work_group_size(1, 1, 1)))
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
#define ERPARTS (MAX_BLOCKSIZE >> 6)
    float len[ERPARTS]; // blocksize / 64!!!!

    __global int *data = &samples[task.data.samplesOffs];
    for (int i = 0; i < ERPARTS; i++)
	len[i] = 0.0f;

    if (ro <= 4)
    {
	float fcoef[4];
	for (int tid = 0; tid < 4; tid++)
	    fcoef[tid] = tid + ro - 4 < 0 ? 0.0f : - ((float) task.coefs[tid + ro - 4]) / (1 << task.data.shift);
	float4 fc0 = vload4(0, &fcoef[0]);
	float fdata[4];
	for (int pos = 0; pos < 4; pos++)
	    fdata[pos] = pos + ro - 4 < 0 ? 0.0f : (float)(data[pos + ro - 4] >> task.data.wbits);
	float4 fd0 = vload4(0, &fdata[0]);
	for (int pos = ro; pos < bs; pos ++)
	{
	    float4 sum4 = fc0 * fd0;
	    float2 sum2 = sum4.s01 + sum4.s23;
	    fd0 = fd0.s1230;
	    fd0.s3 = (float)(data[pos] >> task.data.wbits);
	    len[pos >> 6] += fabs(fd0.s3 + (sum2.x + sum2.y));
	}
    }
    else if (ro <= 8)
    {
	float fcoef[8];
	for (int tid = 0; tid < 8; tid++)
	    fcoef[tid] = tid + ro - 8 < 0 ? 0.0f : - ((float) task.coefs[tid + ro - 8]) / (1 << task.data.shift);
	float8 fc0 = vload8(0, &fcoef[0]);
	float fdata[8];
	for (int pos = 0; pos < 8; pos++)
	    fdata[pos] = pos + ro - 8 < 0 ? 0.0f : (float)(data[pos + ro - 8] >> task.data.wbits);
	float8 fd0 = vload8(0, &fdata[0]);
	for (int pos = ro; pos < bs; pos ++)
	{
	    float8 sum8 = fc0 * fd0;
	    float4 sum4 = sum8.s0123 + sum8.s4567;
	    float2 sum2 = sum4.s01 + sum4.s23;
	    fd0 = fd0.s12345670;
	    fd0.s7 = (float)(data[pos] >> task.data.wbits);
	    len[pos >> 6] += fabs(fd0.s7 + (sum2.x + sum2.y));
	}
    }
    else if (ro <= 12)
    {
	float fcoef[12];
	for (int tid = 0; tid < 12; tid++)
	    fcoef[tid] = tid + ro - 12 >= 0 ? - ((float) task.coefs[tid + ro - 12]) / (1 << task.data.shift) : 0.0f;
	float4 fc0 = vload4(0, &fcoef[0]);
	float4 fc1 = vload4(1, &fcoef[0]);
	float4 fc2 = vload4(2, &fcoef[0]);	
	float fdata[12];
	for (int pos = 0; pos < 12; pos++)
	    fdata[pos] = pos + ro - 12 < 0 ? 0.0f : (float)(data[pos + ro - 12] >> task.data.wbits);
	float4 fd0 = vload4(0, &fdata[0]);
	float4 fd1 = vload4(1, &fdata[0]);
	float4 fd2 = vload4(2, &fdata[0]);
	for (int pos = ro; pos < bs; pos ++)
	{
	    float4 sum4 = fc0 * fd0 + fc1 * fd1 + fc2 * fd2;
	    float2 sum2 = sum4.s01 + sum4.s23;
	    fd0 = fd0.s1230;
	    fd1 = fd1.s1230;
	    fd2 = fd2.s1230;
	    fd0.s3 = fd1.s3;
	    fd1.s3 = fd2.s3;
	    fd2.s3 = (float)(data[pos] >> task.data.wbits);
	    len[pos >> 6] += fabs(fd2.s3 + (sum2.x + sum2.y));
	}
    }
    else
    {
	float fcoef[32];
	for (int tid = 0; tid < 32; tid++)
	    fcoef[tid] = tid < MAX_ORDER && tid + ro - MAX_ORDER >= 0 ? - ((float) task.coefs[tid + ro - MAX_ORDER]) / (1 << task.data.shift) : 0.0f;

	float4 fc0 = vload4(0, &fcoef[0]);
	float4 fc1 = vload4(1, &fcoef[0]);
	float4 fc2 = vload4(2, &fcoef[0]);

	float fdata[MAX_ORDER + TEMPBLOCK1 + 32];
	for (int pos = 0; pos < MAX_ORDER; pos++)
	    fdata[pos] = 0.0f;
	for (int pos = MAX_ORDER + TEMPBLOCK1; pos < MAX_ORDER + TEMPBLOCK1 + 32; pos++)
	    fdata[pos] = 0.0f;
	for (int bpos = 0; bpos < bs; bpos += TEMPBLOCK1)
	{
	    int end = min(bpos + TEMPBLOCK1, bs);

	    for (int pos = max(bpos - ro, 0); pos < max(bpos, ro); pos++)
		fdata[MAX_ORDER + pos - bpos] = (float)(data[pos] >> task.data.wbits);	

	    for (int pos = max(bpos, ro); pos < end; pos ++)
	    {
		float next = (float)(data[pos] >> task.data.wbits);
		float * dptr = fdata + pos - bpos;
		dptr[MAX_ORDER] = next;
		float4 sum 
		    = fc0 * vload4(0, dptr)
		    + fc1 * vload4(1, dptr)
    #if MAX_ORDER > 8
		    + fc2 * vload4(2, dptr)
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
		next += sum.x + sum.y + sum.z + sum.w;
		len[pos >> 6] += fabs(next);
	    }
	}
    }

    int total = 0;
    for (int i = 0; i < ERPARTS; i++)
    {
	int res = convert_int_sat_rte(len[i] * 2);
	int k = clamp(31 - clz(res) - 6, 0, MAX_RICE_PARAM); // 25 - clz(res) == clz(64) - clz(res) == log2(res / 64)
	total += (k << 6) + (res >> k);
    }
    int partLen = min(0x7ffffff, total) + (bs - ro);
    int obits = task.data.obits - task.data.wbits;
    tasks[selectedTask].data.size = min(obits * bs,
	task.data.type == Fixed ? ro * obits + 6 + RICE_PARAM_BITS + partLen :
	task.data.type == LPC ? ro * obits + 4 + 5 + ro * task.data.cbits + 6 + RICE_PARAM_BITS/* << porder */ + partLen :
	task.data.type == Constant ? obits * select(1, bs, partLen != bs - ro) :
	obits * bs);
}
#else
#define ESTPARTLOG 5

__kernel /*__attribute__(( vec_type_hint (int4)))*/ __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clEstimateResidual(
    __global int*samples,
    __global int*selectedTasks,
    __global FLACCLSubframeTask *tasks
    )
{
    __local float data[GROUP_SIZE * 2 + 32];
#if !defined(AMD)
    __local volatile uint idata[GROUP_SIZE + 16];
#endif
    __local FLACCLSubframeTask task;
    __local uint psum[MAX_BLOCKSIZE >> ESTPARTLOG];
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
    for (int offs = tid; offs < (MAX_BLOCKSIZE >> ESTPARTLOG); offs += GROUP_SIZE)
	psum[offs] = 0;
    data[tid] = 0.0f;
    // need to initialize "extra" data, because NaNs can produce weird results even when multiplied by zero extra coefs
    if (tid < 32)
	data[GROUP_SIZE * 2 + tid] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    float4 fc0 = vload4(0, &fcoef[0]);
    float4 fc1 = vload4(1, &fcoef[0]);
#if MAX_ORDER > 8
    float4 fc2 = vload4(2, &fcoef[0]);
#endif
    __global int * rptr = &samples[task.data.samplesOffs];
    int wb = task.data.wbits;
    int pos;
    for (pos = 0; pos + GROUP_SIZE - 1 < bs; pos += GROUP_SIZE)
    {
	// fetch samples
	int offs = pos + tid;
	float nextData = rptr[offs] >> wb;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	// compute residual
	__local float* dptr = &data[tid + GROUP_SIZE - MAX_ORDER];
	float4 sum4
	    = fc0 * vload4(0, dptr)
	    + fc1 * vload4(1, dptr)
#if MAX_ORDER > 8
	    + fc2 * vload4(2, dptr)
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

	float2 sum2 = sum4.s01 + sum4.s23;
	int it = convert_int_sat_rte(nextData + (sum2.s0 + sum2.s1));
//	int t = (int)(nextData + sum.x + sum.y + sum.z + sum.w);
	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
	// convert to unsigned
	uint t = (it << 1) ^ (it >> 31);
	// ensure we're within frame bounds
	t = select(0U, t, offs >= ro);
	// overflow protection
	t = min(t, 0x7ffffffU);
#if defined(AMD)
	atomic_add(&psum[min(MAX_BLOCKSIZE - 1, offs) >> ESTPARTLOG], t);
#else
	idata[tid] = t;
#if WARP_SIZE <= (1 << (ESTPARTLOG - 1))
        barrier(CLK_LOCAL_MEM_FENCE);
	for (int l = 1 << (ESTPARTLOG - 1); l >= WARP_SIZE; l >>= 1) {
            if (!(tid & l)) idata[tid] += idata[tid + l];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
	for (int l = WARP_SIZE / 2; l > 1; l >>= 1)
	    idata[tid] += idata[tid + l];
#else
	for (int l = 1 << (ESTPARTLOG - 1); l > 1; l >>= 1)
	    idata[tid] += idata[tid + l];
#endif
	if ((tid & (1 << ESTPARTLOG) - 1) == 0)
            psum[min(MAX_BLOCKSIZE - 1, offs) >> ESTPARTLOG] = idata[tid] + idata[tid + 1];
#endif
    }
    if (pos < bs)
    {
	// fetch samples
	int offs = pos + tid;
	float nextData = offs < bs ? rptr[offs] >> wb : 0.0f;
	data[tid + GROUP_SIZE] = nextData;
	barrier(CLK_LOCAL_MEM_FENCE);

	// compute residual
	__local float* dptr = &data[tid + GROUP_SIZE - MAX_ORDER];
	float4 sum 
	    = fc0 * vload4(0, dptr)
	    + fc1 * vload4(1, dptr)
#if MAX_ORDER > 8
	    + fc2 * vload4(2, dptr)
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

	int it = convert_int_sat_rte(nextData + sum.x + sum.y + sum.z + sum.w);
	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
	// convert to unsigned
	uint t = (it << 1) ^ (it >> 31);
	// ensure we're within frame bounds
	t = select(0U, t, offs >= ro && offs < bs);
	// overflow protection
	t = min(t, 0x7ffffffU);
#if defined(AMD)
	atomic_add(&psum[min(MAX_BLOCKSIZE - 1, offs) >> ESTPARTLOG], t);
#else
	idata[tid] = t;
#if WARP_SIZE <= (1 << (ESTPARTLOG - 1))
        barrier(CLK_LOCAL_MEM_FENCE);
	for (int l = 1 << (ESTPARTLOG - 1); l >= WARP_SIZE; l >>= 1) {
            if (!(tid & l)) idata[tid] += idata[tid + l];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
	for (int l = WARP_SIZE / 2; l > 1; l >>= 1)
	    idata[tid] += idata[tid + l];
#else
	for (int l = 1 << (ESTPARTLOG - 1); l > 1; l >>= 1)
	    idata[tid] += idata[tid + l];
#endif
	if ((tid & (1 << ESTPARTLOG) - 1) == 0)
	    psum[min(MAX_BLOCKSIZE - 1, offs) >> ESTPARTLOG] = idata[tid] + idata[tid + 1];
#endif
    }

    // calculate rice partition bit length for every 32 samples
    barrier(CLK_LOCAL_MEM_FENCE);
#if (MAX_BLOCKSIZE >> (ESTPARTLOG + 1)) > GROUP_SIZE
#error MAX_BLOCKSIZE is too large for this GROUP_SIZE
#endif
    uint pl = tid < (MAX_BLOCKSIZE >> (ESTPARTLOG + 1)) ? psum[tid * 2] + psum[tid * 2 + 1] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
 //   for (int pos = 0; pos < (MAX_BLOCKSIZE >> ESTPARTLOG) / 2; pos += GROUP_SIZE)
 //   {
	//int offs = pos + tid;
	//int pl = offs < (MAX_BLOCKSIZE >> ESTPARTLOG) / 2 ? psum[offs * 2] + psum[offs * 2 + 1] : 0;
	////int pl = psum[offs * 2] + psum[offs * 2 + 1];
	//barrier(CLK_LOCAL_MEM_FENCE);
	//if (offs < (MAX_BLOCKSIZE >> ESTPARTLOG) / 2)
	//    psum[offs] = pl;
 //   }
    int k = clamp(31 - (int)clz(pl) - (ESTPARTLOG + 1), 0, MAX_RICE_PARAM); // 26 - clz(res) == clz(32) - clz(res) == log2(res / 32)
    if (tid < MAX_BLOCKSIZE >> (ESTPARTLOG + 1))
	psum[tid] = (k << (ESTPARTLOG + 1)) + (pl >> k);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int l = MAX_BLOCKSIZE >> (ESTPARTLOG + 2); l > 0; l >>= 1)
    {
	if (tid < l)
	    psum[tid] += psum[tid + l];
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (tid == 0)
    {
	int pl = (int)psum[0] + (bs - ro);
	int obits = task.data.obits - task.data.wbits;
	int len = min(obits * task.data.blocksize,
		task.data.type == Fixed ? task.data.residualOrder * obits + 6 + RICE_PARAM_BITS + pl :
		task.data.type == LPC ? task.data.residualOrder * obits + 4 + 5 + task.data.residualOrder * task.data.cbits + 6 + RICE_PARAM_BITS/* << porder */ + pl :
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
    int tasksWindow,
    int windowCount,
    int tasksToSecondEstimate,
    int taskCount,
    int selectedCount
    )
{
    int best_size[4];
    int best_wind[4];
    for (int ch = 0; ch < 4; ch++)
    {
	int first_no = selectedTasks[(get_global_id(0) * 4 + ch) * selectedCount];
	int best_len = tasks[first_no].data.size;
	int best_wnd = 0;
	for (int i = 1; i < selectedCount; i++)
	{
	    int task_no = selectedTasks[(get_global_id(0) * 4 + ch) * selectedCount + i];
	    int task_len = tasks[task_no].data.size;
	    int task_wnd = (task_no - first_no) / tasksWindow;
	    task_wnd = select(0, task_wnd, task_wnd < windowCount);
	    best_wnd = select(best_wnd, task_wnd, task_len < best_len);
	    best_len = min(task_len, best_len);
	}
	best_size[ch] = best_len;
	best_wind[ch] = best_wnd;
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
	for (int j = taskCount - 1; j >= 0; j--)
	{
	    int i = select(j, (j % windowCount) * tasksWindow + (j / windowCount), j < windowCount * tasksWindow);
	    int no = (get_global_id(0) * 4 + ch) * taskCount + i;
	    selectedTasksBestMethod[(get_global_id(0) * 2 + ich) * taskCount + i] = no;
	    tasks[no].data.residualOffs = roffs;
	    if (j >= selectedCount)
		tasks[no].data.size = 0x7fffffff;
	    if (nonSelectedNo < tasksToSecondEstimate)
		if (tasksToSecondEstimate == taskCount - selectedCount || best_wind[ch] == i / tasksWindow || i >= windowCount * tasksWindow)
		    selectedTasksSecondEstimate[(get_global_id(0) * 2 + ich) * tasksToSecondEstimate + nonSelectedNo++] = no;
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

#if BITS_PER_SAMPLE > 16
#define residual_t long
#define convert_bps_sat convert_int_sat
#else
#define residual_t int
#define convert_bps_sat
#endif

#ifdef FLACCL_CPU
inline residual_t calc_residual(__global int *ptr, int * coefs, int ro)
{
    residual_t sum = 0;
    for (int i = 0; i < ro; i++)
	sum += (residual_t)ptr[i] * coefs[i];
        //sum += upsample(mul_hi(ptr[i], coefs[i]), as_uint(ptr[i] * coefs[i]));
    return sum;
}

#define ENCODE_N(cro,action) for (int pos = cro; pos < bs; pos ++) { \
	residual_t t = (data[pos] - (calc_residual(data + pos - cro, task.coefs, cro) >> task.data.shift)) >> task.data.wbits; \
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

// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clEncodeResidual(
    __global ulong *partition_lengths,
    __global int *residual,
    __global int *samples,
    __global FLACCLSubframeTask *tasks,
    int max_porder, // <= 8
    int psize // == task.blocksize >> max_porder?
    )
{
    FLACCLSubframeTask task = tasks[get_group_id(0)];
    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;
    __global int *data = &samples[task.data.samplesOffs];
    __global ulong *pl = partition_lengths + (1 << (max_porder + 1)) * get_group_id(0);
    int r;
    for (int p = 0; p < (1 << max_porder); p++)
	pl[p] = 0UL;
    __global int *rptr = residual + task.data.residualOffs;
    if (psize == 16)
    {
	SWITCH_N((rptr[pos] = r = convert_bps_sat(t), pl[pos >> 4] += (uint)((r << 1) ^ (r >> 31))));
    }
    else
    {
	SWITCH_N((rptr[pos] = r = convert_bps_sat(t), pl[pos / psize] += (uint)((r << 1) ^ (r >> 31))));
    }
}
#else
// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clEncodeResidual(
    __global int *partition_lengths,
    __global int *output,
    __global int *samples,
    __global FLACCLSubframeTask *tasks,
    int max_porder, // <= 8
    int psize // == task.blocksize >> max_porder?
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

    int4 cptr0 = vload4(0, &task.coefs[0]);
    int4 cptr1 = vload4(1, &task.coefs[0]);
#if MAX_ORDER > 8
    int4 cptr2 = vload4(2, &task.coefs[0]);
#endif

    // We tweaked coeffs so that (task.cbits + task.obits + log2i(ro) <= 32) 
    // when BITS_PER_SAMPLE == 16, so we don't need 64bit arithmetics.

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
#if BITS_PER_SAMPLE > 16
	long4 sum
            = upsample(mul_hi(cptr0, vload4(0, dptr)), as_uint4(cptr0 * vload4(0, dptr)))
            + upsample(mul_hi(cptr1, vload4(1, dptr)), as_uint4(cptr1 * vload4(1, dptr)))
  #if MAX_ORDER > 8
            + upsample(mul_hi(cptr2, vload4(2, dptr)), as_uint4(cptr2 * vload4(2, dptr)))
    #if MAX_ORDER > 12
            + upsample(mul_hi(vload4(3, &task.coefs[0]), vload4(3, dptr)), as_uint4(vload4(3, &task.coefs[0]) * vload4(3, dptr)))
      #if MAX_ORDER > 16
            + upsample(mul_hi(vload4(4, &task.coefs[0]), vload4(4, dptr)), as_uint4(vload4(4, &task.coefs[0]) * vload4(4, dptr)))
            + upsample(mul_hi(vload4(5, &task.coefs[0]), vload4(5, dptr)), as_uint4(vload4(5, &task.coefs[0]) * vload4(5, dptr)))
            + upsample(mul_hi(vload4(6, &task.coefs[0]), vload4(6, dptr)), as_uint4(vload4(6, &task.coefs[0]) * vload4(6, dptr)))
            + upsample(mul_hi(vload4(7, &task.coefs[0]), vload4(7, dptr)), as_uint4(vload4(7, &task.coefs[0]) * vload4(7, dptr)))
      #endif
    #endif
  #endif
#else
	int4 sum 
            = cptr0 * vload4(0, dptr)
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
	    output[task.data.residualOffs + off] = convert_bps_sat(nextData - ((sum.x + sum.y + sum.z + sum.w) >> task.data.shift));

	barrier(CLK_LOCAL_MEM_FENCE);
	data[tid] = nextData;
    }
}
#endif

#ifndef FLACCL_CPU
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
    __local uint pl[(GROUP_SIZE / 16)][MAX_RICE_PARAM + 1];
    __local FLACCLSubframeData task;

    const int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(1)]))[tid];
    if (tid < (GROUP_SIZE / 16))
    {
	for (int k = 0; k <= MAX_RICE_PARAM; k++)
	    pl[tid][k] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int start = get_group_id(0) * psize * (GROUP_SIZE / 16);
    int end = min(start + psize * (GROUP_SIZE / 16), task.blocksize);
    for (int offs = start + tid; offs < end; offs += GROUP_SIZE)
    {
	// fetch residual
	int s = (offs >= task.residualOrder && offs < end) ? residual[task.residualOffs + offs] : 0;
	// convert to unsigned
	uint t = (s << 1) ^ (s >> 31);
	// calc number of unary bits for each residual sample with each rice parameter
	int part = (offs - start) / psize;
	// we must ensure that psize * (t >> k) doesn't overflow;
	uint lim = 0x7fffffffU / (uint)psize;
	for (int k = 0; k <= MAX_RICE_PARAM; k++)
	    atomic_add(&pl[part][k], min(lim, t >> k));
	    //pl[part][k] += s >> k;
    }   
    barrier(CLK_LOCAL_MEM_FENCE);

    int part = get_group_id(0) * (GROUP_SIZE / 16) + tid;
    if (tid < (GROUP_SIZE / 16) && part < (1 << max_porder))
    {
	for (int k = 0; k <= MAX_RICE_PARAM; k++)
	{
	    // output length
	    const int pos = ((MAX_RICE_PARAM + 1) << (max_porder + 1)) * get_group_id(1) + (k << (max_porder + 1));
	    uint plen = pl[tid][k];
	    partition_lengths[pos + part] = min(0x007fffffU, plen) + (uint)(psize - select(0, task.residualOrder, part == 0)) * (k + 1);
	 //   if (get_group_id(1) == 0)
		//printf("pl[%d][%d] == %d\n", k, part, min(0x7fffff, pl[k][tid]) + (psize - task.residualOrder * (part == 0)) * (k + 1));
	}
    }
}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clCalcPartition16(
    __global unsigned int *partition_lengths,
    __global int *residual,
    __global FLACCLSubframeTask *tasks,
    int max_porder // <= 8
    )
{
    __local FLACCLSubframeData task;
    __local unsigned int res[GROUP_SIZE];
    __local unsigned int pl[GROUP_SIZE >> 4][MAX_RICE_PARAM + 1];

    const int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0)]))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    int bs = task.blocksize;
    int ro = task.residualOrder;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	int offs = pos + tid;
	// fetch residual
	int s = (offs >= ro && offs < bs) ? residual[task.residualOffs + offs] : 0;
	// convert to unsigned
	res[tid] = (s << 1) ^ (s >> 31);

	barrier(CLK_LOCAL_MEM_FENCE);

	// we must ensure that psize * (t >> k) doesn't overflow;
	uint4 lim = 0x07ffffffU;
	int x = tid >> 4;
	__local uint * chunk = &res[x << 4];
	for (int k0 = 0; k0 <= MAX_RICE_PARAM; k0 += 16)
	{
	    // calc number of unary bits for each group of 16 residual samples
	    // with each rice parameter.
	    int k = k0 + (tid & 15);
	    uint4 rsum 
		= min(lim, vload4(0,chunk) >> k)
		+ min(lim, vload4(1,chunk) >> k)
		+ min(lim, vload4(2,chunk) >> k)
		+ min(lim, vload4(3,chunk) >> k)
		;
	    uint rs = rsum.x + rsum.y + rsum.z + rsum.w;

	    // We can safely limit length here to 0x007fffffU, not causing length
	    // mismatch, because any such length would cause Verbatim frame anyway.
	    // And this limit protects us from overflows when calculating larger 
	    // partitions, as we can have a maximum of 2^8 partitions, resulting
	    // in maximum partition length of 0x7fffffffU + change.
	    if (k <= MAX_RICE_PARAM) pl[x][k] = min(0x007fffffU, rs) + (uint)(16 - select(0, ro, offs < 16)) * (k + 1);
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int k0 = 0; k0 <= MAX_RICE_PARAM; k0 += 16)
	{
            int k1 = k0 + (tid >> (GROUP_SIZE_LOG - 4)), x1 = tid & ((1 << (GROUP_SIZE_LOG - 4)) - 1);
	    if (k1 <= MAX_RICE_PARAM && (pos >> 4) + x1 < (1 << max_porder))
		partition_lengths[((MAX_RICE_PARAM + 1) << (max_porder + 1)) * get_group_id(0) + (k1 << (max_porder + 1)) + (pos >> 4) + x1] = pl[x1][k1];
	}
    }    
}
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clCalcPartition32(
    __global unsigned int *partition_lengths,
    __global int *residual,
    __global FLACCLSubframeTask *tasks,
    int max_porder // <= 8
    )
{
    __local FLACCLSubframeData task;
    __local unsigned int res[GROUP_SIZE];
    __local unsigned int pl[GROUP_SIZE >> 5][32];
    const int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0)]))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    int bs = task.blocksize;
    int ro = task.residualOrder;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int pos = 0; pos < bs; pos += GROUP_SIZE)
    {
	int offs = pos + tid;
	// fetch residual
	int s = (offs >= ro && offs < bs) ? residual[task.residualOffs + offs] : 0;
	// convert to unsigned
	res[tid] = (s << 1) ^ (s >> 31);

	barrier(CLK_LOCAL_MEM_FENCE);

	// we must ensure that psize * (t >> k) doesn't overflow;
	uint4 lim = 0x03ffffffU;
	int x = tid >> 5;
	__local uint * chunk = &res[x << 5];
	// calc number of unary bits for each group of 32 residual samples
	// with each rice parameter.
	int k = tid & 31;
	uint4 rsum 
	    = min(lim, vload4(0,chunk) >> k)
	    + min(lim, vload4(1,chunk) >> k)
	    + min(lim, vload4(2,chunk) >> k)
	    + min(lim, vload4(3,chunk) >> k)
	    + min(lim, vload4(4,chunk) >> k)
	    + min(lim, vload4(5,chunk) >> k)
	    + min(lim, vload4(6,chunk) >> k)
	    + min(lim, vload4(7,chunk) >> k)
	    ;
	uint rs = rsum.x + rsum.y + rsum.z + rsum.w;

	// We can safely limit length here to 0x007fffffU, not causing length
	// mismatch, because any such length would cause Verbatim frame anyway.
	// And this limit protects us from overflows when calculating larger 
	// partitions, as we can have a maximum of 2^8 partitions, resulting
	// in maximum partition length of 0x7fffffffU + change.
	if (k <= MAX_RICE_PARAM) pl[x][k] = min(0x007fffffU, rs) + (uint)(32 - select(0, ro, offs < 32)) * (k + 1);

	barrier(CLK_LOCAL_MEM_FENCE);
	
	int k1 = (tid >> (GROUP_SIZE_LOG - 5)), x1 = tid & ((1 << (GROUP_SIZE_LOG - 5)) - 1);
	if (k1 <= MAX_RICE_PARAM && (pos >> 5) + x1 < (1 << max_porder))
	    partition_lengths[((MAX_RICE_PARAM + 1) << (max_porder + 1)) * get_group_id(0) + (k1 << (max_porder + 1)) + (pos >> 5) + x1] = pl[x1][k1];
    }
}
#endif

#ifdef FLACCL_CPU
// Sums partition lengths for a certain k == get_group_id(0)
// get_group_id(0) == k
// get_group_id(1) == task index
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clSumPartition(
    __global ulong* partition_lengths,
    int max_porder
    )
{
    if (get_group_id(0) != 0) // ignore k != 0
	return;
    __global ulong * sums = partition_lengths + (1 << (max_porder + 1)) * get_group_id(1);
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
    __global uint* partition_lengths,
    int max_porder
    )
{
    __local uint data[256]; // max_porder <= 8, data length <= 1 << 9.
    const int pos = ((MAX_RICE_PARAM + 1) << (max_porder + 1)) * get_group_id(1) + (get_group_id(0) << (max_porder + 1));

    // fetch partition lengths
    uint2 pl = get_local_id(0) * 2 < (1 << max_porder) ? vload2(get_local_id(0),&partition_lengths[pos]) : 0;
    data[get_local_id(0)] = pl.x + pl.y;
    barrier(CLK_LOCAL_MEM_FENCE);

    int in_pos = (get_local_id(0) << 1);
    int out_pos = (1 << (max_porder - 1)) + get_local_id(0);
    for (int bs = 1 << (max_porder - 2); bs > 0; bs >>= 1)
    {
	if (get_local_id(0) < bs) data[out_pos] =  data[in_pos] + data[in_pos + 1];
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

#ifdef FLACCL_CPU
// Finds optimal rice parameter for each partition.
// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clFindRiceParameter(
    __global FLACCLSubframeTask *tasks,
    __global int* rice_parameters,
    __global ulong* partition_lengths,
    int max_porder
    )
{
    __global FLACCLSubframeTask* task = tasks + get_group_id(0);
    const int tid = get_local_id(0);
    int lim = (2 << max_porder) - 1;
    //int psize = task->data.blocksize >> max_porder;
    int bs = task->data.blocksize;
    int ro = task->data.residualOrder;
    __global ulong* ppl = &partition_lengths[get_group_id(0) << (max_porder + 1)];
    __global int* prp = &rice_parameters[get_group_id(0) << (max_porder + 2)];
    __global int* pol = prp + (1 << (max_porder + 1));
    for (int porder = max_porder; porder >= 0; porder--)
    {
	int pos = (2 << max_porder) - (2 << porder);
	int fin = pos + (1 << porder);

	ulong pl = ppl[pos];
	int ps = (bs >> porder) - ro;
	int k = clamp(63 - (int)clz(pl / max(1, ps)), 0, MAX_RICE_PARAM);
	int plk = ps * (k + 1) + (int)(pl >> k);

	// output rice parameter
	prp[pos] = k;
	// output length
	pol[pos] = plk;

	ps = (bs >> porder);

	for (int offs = pos + 1; offs < fin; offs++)
	{
	    pl = ppl[offs];
	    k = clamp(63 - (int)clz(pl / ps), 0, MAX_RICE_PARAM);
	    plk = ps * (k + 1) + (int)(pl >> k);
	
	    // output rice parameter
	    prp[offs] = k;
	    // output length
	    pol[offs] = plk;
	}
    }
}
#else
// Finds optimal rice parameter for each partition.
// get_group_id(0) == task index
__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void clFindRiceParameter(
    __global FLACCLSubframeTask *tasks,
    __global int* rice_parameters,
    __global uint* partition_lengths,
    int max_porder
    )
{
    for (int offs = get_local_id(0); offs < (2 << max_porder); offs += GROUP_SIZE)
    {
	const int pos = ((MAX_RICE_PARAM + 1) << (max_porder + 1)) * get_group_id(0) + offs;
	uint best_l = partition_lengths[pos];
	int best_k = 0;
	for (int k = 1; k <= MAX_RICE_PARAM; k++)
	{
	    uint l = partition_lengths[pos + (k << (max_porder + 1))];
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

#ifdef FLACCL_CPU
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
    
    for (int porder = max_porder; porder >= 0; porder--)
    {
	int start = (2 << max_porder) - (2 << porder);
	for (int offs = 0; offs < (1 << porder); offs ++)
	    partlen[porder] += rice_parameters[pos + start + offs];
    }

    int best_length = partlen[0] + RICE_PARAM_BITS;
    int best_porder = 0;
    for (int porder = 1; porder <= max_porder; porder++)
    {
	int length = (RICE_PARAM_BITS << porder) + partlen[porder];
	best_porder = select(best_porder, porder, length < best_length);
	best_length = min(best_length, length);
    }

    best_length = (RICE_PARAM_BITS << best_porder) + task->data.blocksize - task->data.residualOrder;
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
	atomic_add(&partlen[porder], len);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int best_length = partlen[0] + RICE_PARAM_BITS;
    int best_porder = 0;
    for (int porder = 1; porder <= max_porder; porder++)
    {
	int length = (RICE_PARAM_BITS << porder) + partlen[porder];
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
#ifdef FLACCL_CPU
typedef struct BitWriter_t
{
    __global unsigned int *buffer;
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
    return select(8, 8 * ((bts + 4) / 5), bts > 6);
}

#ifdef FLACCL_CPU
// get_global_id(0) * channels == task index
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
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
	offset += select(0, select(8, 16, bs >= 256), bs != 4096 && bs != 4608); // TODO: check all other standard sizes
	
	// assert (offset % 8) == 0
	offset += 8;
	for (int ch = 0; ch < channels; ch++)
	{
	    __global FLACCLSubframeTask* task = tasks + iFrame * channels + ch;
	    offset += 8 + task->data.wbits;
	    // Add 32 bits to separate frames if header is too small so they can intersect
	    offset += 64;
	    task->data.encodingOffset = offset + task->data.headerLen;
	    offset += task->data.size;
	}
	offset = (offset + 7) & ~7;
	offset += 16;
    }
}
#else
// get_global_id(0) * channels == task index
__kernel __attribute__((reqd_work_group_size(32, 1, 1)))
void clCalcOutputOffsets(
    __global int *residual,
    __global int *samples,
    __global FLACCLSubframeTask *tasks,
    int channels1,
    int frameCount,
    int firstFrame
    )
{
    __local FLACCLSubframeData ltasks[MAX_CHANNELS];
    __local volatile int mypos[MAX_CHANNELS];
    int offset = 0;
    for (int iFrame = 0; iFrame < frameCount; iFrame++)
    {
	if (get_local_id(0) < sizeof(ltasks[0]) / sizeof(int))
	    for (int ch = 0; ch < MAX_CHANNELS; ch++)
		((__local int*)&ltasks[ch])[get_local_id(0)] = ((__global int*)(&tasks[iFrame * MAX_CHANNELS + ch]))[get_local_id(0)];

	//printf("len_utf8(%d) == %d\n", firstFrame + iFrame, len_utf8(firstFrame + iFrame));
	offset += 15 + 1 + 4 + 4 + 4 + 3 + 1 + len_utf8(firstFrame + iFrame)
	    // + 8-16 // custom block size
	    // + 8-16 // custom sample rate
	    ;
	int bs = ltasks[0].blocksize;
	//public static readonly int[] flac_blocksizes = new int[15] { 0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384 };	
	offset += select(0, select(8, 16, bs >= 256), bs != 4096 && bs != 4608); // TODO: check all other standard sizes
	
	// assert (offset % 8) == 0
	offset += 8;
	if (get_local_id(0) < MAX_CHANNELS)
	{
	    int ch = get_local_id(0);
	    // Add 64 bits to separate frames if header is too small so they can intersect
	    int mylen = 8 + ltasks[ch].wbits + 64 + ltasks[ch].size;
	    mypos[ch] = mylen;
	    for (int offset = 1; offset < WARP_SIZE && offset < MAX_CHANNELS; offset <<= 1)
		if (ch >= offset) mypos[ch] += mypos[ch - offset];
	    mypos[ch] += offset;
	    tasks[iFrame * MAX_CHANNELS + ch].data.encodingOffset = mypos[ch] - ltasks[ch].size + ltasks[ch].headerLen;
	}
	offset = mypos[MAX_CHANNELS - 1];
	offset = (offset + 7) & ~7;
	offset += 16;
    }
}
#endif

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
#ifdef FLACCL_CPU
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
	    writebits(&bw, RICE_PARAM_BITS, k);
	    //if (get_group_id(0) == 0) printf("[%x] ", k);
	    //if (get_group_id(0) == 0) printf("(%x) ", bw.bit_buf);
	    if (p == 1) res_cnt = psize;
	    int cnt = min(res_cnt, bs - j);
	    unsigned int kexp = 1U << k;
	    __global int *rptr = &residual[task->data.residualOffs + j];
	    for (int i = 0; i < cnt; i++)
	    {
		int iv = rptr[i];
		unsigned int v = (iv << 1) ^ (iv >> 31);
		// write quotient in unary
		int bits = k + (v >> k) + 1;
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
			bw.buffer[bw.buf_ptr++] = as_int(as_char4(bb).wzyx);
		    }
		    bits -= b;
		}
		unsigned int val = (v & (kexp - 1)) | kexp;
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
		    bw.buffer[bw.buf_ptr++] = as_int(as_char4(bb).wzyx);
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
    __local uint data[GROUP_SIZE];
    __local volatile int mypos[GROUP_SIZE+1];
#if 0
    __local int brp[256];
#endif
    __local volatile int warppos[WARP_SIZE];
    __local FLACCLSubframeData task;

    int tid = get_local_id(0);
    if (tid < sizeof(task) / sizeof(int))
	((__local int*)&task)[tid] = ((__global int*)(&tasks[get_group_id(0)]))[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (task.type != Fixed && task.type != LPC)
	return;
    if (tid == 0)
	mypos[GROUP_SIZE] = 0;
    if (tid < WARP_SIZE)
	warppos[tid] = 0;
#if 0
    for (int offs = tid; offs < (1 << task.porder); offs ++)
	brp[offs] = best_rice_parameters[(get_group_id(0) << max_porder) + offs];
#endif
    data[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    const int bs = task.blocksize;
    int start = task.encodingOffset;
    int plen = bs >> task.porder;
    //int plenoffs = 12 - task.porder;
    uint remainder = 0U;
    int pos;
    for (pos = 0; pos + GROUP_SIZE - 1 < bs; pos += GROUP_SIZE)
    {
	int offs = pos + tid;
	int iv = residual[task.residualOffs + offs];
	int part = offs / plen;
	//int part = offs >> plenoffs;
#if 0
	int k = brp[part];
#else
	int k = best_rice_parameters[(get_group_id(0) << max_porder) + part];
#endif
	int pstart = offs == part * plen;
        //int pstart = offs == part << plenoffs;
	uint v = (iv << 1) ^ (iv >> 31);
	int mylen = select(0, (int)(v >> k) + 1 + k, offs >= task.residualOrder && offs < bs) + select(0, RICE_PARAM_BITS, pstart);
	mypos[tid] = mylen;

	// Inclusive scan(+)
	int lane = (tid & (WARP_SIZE - 1));
	for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
	    mypos[tid] += mypos[select(GROUP_SIZE, tid - offset, lane >= offset)];
	int mp = mypos[tid];
	if ((tid & (WARP_SIZE - 1)) == WARP_SIZE - 1)
	    warppos[tid/WARP_SIZE] = mp;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (tid < GROUP_SIZE/WARP_SIZE)
	{
	    for (int offset = 1; offset < GROUP_SIZE/WARP_SIZE; offset <<= 1)
		warppos[tid] += warppos[select(GROUP_SIZE/WARP_SIZE, tid - offset, tid >= offset)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	mp += start + select(0, warppos[tid / WARP_SIZE - 1], tid / WARP_SIZE > 0);
	int start32 = start >> 5;
	start += mypos[GROUP_SIZE - 1] + warppos[GROUP_SIZE / WARP_SIZE - 2];
	//if (start / 32 - start32 >= GROUP_SIZE - 3)
	//    tasks[get_group_id(0)].data.size = 1;
	//if (tid == GROUP_SIZE - 1 && mypos[tid] > (GROUP_SIZE/2) * 32)
	//    printf("Oops: %d\n", mypos[tid]);
	data[tid] = select(0U, remainder, tid == 0);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (pstart)
	{
	    int kpos = mp - mylen;
	    int kpos0 = (kpos >> 5) - start32;
	    int kpos1 = kpos & 31;
	    uint kval = (uint)k << (32 - RICE_PARAM_BITS);
	    uint kval0 = kval >> kpos1;
	    uint kval1 = kval << (32 - kpos1);
	    if (kval0) atomic_or(&data[kpos0], kval0);
	    if (kpos1 && kval1) atomic_or(&data[kpos0 + 1], kval1);
	}
        if (offs >= task.residualOrder && offs < bs)
        {
	    int qpos = mp - k - 1;
	    int qpos0 = (qpos >> 5) - start32;
	    int qpos1 = qpos & 31;
	    uint qval = (1U << 31) | (v << (31 - k));
	    uint qval0 = qval >> qpos1;
	    uint qval1= qval << (32 - qpos1);
	    if (qval0) atomic_or(&data[qpos0], qval0);
	    if (qpos1 && qval1) atomic_or(&data[qpos0 + 1], qval1);
        }
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((start32 + tid) * 32 <= start)
	    output[start32 + tid] = as_int(as_char4(data[tid]).wzyx);
	remainder = data[start / 32 - start32];
    }
    if (pos < bs)
    {
	int offs = pos + tid;
	int iv = offs < bs ? residual[task.residualOffs + offs] : 0;
	int part = offs / plen; // >> plenoffs;
	//int k = brp[min(255, part)];
	int k = offs < bs ? best_rice_parameters[(get_group_id(0) << max_porder) + part] : 0;
	int pstart = offs == part * plen;
	uint v = (iv << 1) ^ (iv >> 31);
	int mylen = select(0, (int)(v >> k) + 1 + k, offs >= task.residualOrder && offs < bs) + select(0, RICE_PARAM_BITS, pstart);
	mypos[tid] = mylen;
	
	// Inclusive scan(+)
	int lane = (tid & (WARP_SIZE - 1));
	for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
	    mypos[tid] += mypos[select(GROUP_SIZE, tid - offset, lane >= offset)];
	int mp = mypos[tid];
	if ((tid & (WARP_SIZE - 1)) == WARP_SIZE - 1)
	    warppos[tid/WARP_SIZE] = mp;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (tid < GROUP_SIZE/WARP_SIZE)
	{
	    for (int offset = 1; offset < GROUP_SIZE/WARP_SIZE; offset <<= 1)
		warppos[tid] += warppos[select(GROUP_SIZE/WARP_SIZE, tid - offset, tid >= offset)];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	mp += start + select(0, warppos[tid / WARP_SIZE - 1], tid / WARP_SIZE > 0);
	int start32 = start >> 5;
	start += mypos[GROUP_SIZE - 1] + warppos[GROUP_SIZE / WARP_SIZE - 2];

	//if (tid == GROUP_SIZE - 1 && mypos[tid] > (GROUP_SIZE/2) * 32)
	//    printf("Oops: %d\n", mypos[tid]);
	data[tid] = select(0U, remainder, tid == 0);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (pstart)
	{
	    int kpos = mp - mylen;
	    int kpos0 = (kpos >> 5) - start32;
	    int kpos1 = kpos & 31;
	    uint kval = (uint)k << (32 - RICE_PARAM_BITS);
	    uint kval0 = kval >> kpos1;
	    uint kval1 = kval << (32 - kpos1);
	    if (kval0) atomic_or(&data[kpos0], kval0);
	    if (kpos1 && kval1) atomic_or(&data[kpos0 + 1], kval1);
	}
        if (offs >= task.residualOrder && offs < bs)
        {
	    int qpos = mp - k - 1;
	    int qpos0 = (qpos >> 5) - start32;
	    int qpos1 = qpos & 31;
	    uint qval = (1U << 31) | (v << (31 - k));
	    uint qval0 = qval >> qpos1;
	    uint qval1= qval << (32 - qpos1);
	    if (qval0) atomic_or(&data[qpos0], qval0);
	    if (qpos1 && qval1) atomic_or(&data[qpos0 + 1], qval1);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((start32 + tid) * 32 <= start)
	    output[start32 + tid] = as_int(as_char4(data[tid]).wzyx);
	remainder = data[start / 32 - start32];
    }
    //   if (tid == 0 && start != task.encodingOffset - task.headerLen + task.size)
	//printf("size mismatch: %d != %d\n", start, task.encodingOffset - task.headerLen + task.size);
#endif
}
#endif /* DO_RICE */
#endif /* DO_PARTITIONS */
#endif
