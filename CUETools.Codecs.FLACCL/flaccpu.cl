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

#ifdef DEBUG
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

#pragma OPENCL EXTENSION cl_amd_fp64 : enable

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
    }
}

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
    float data1[4096 + 32];

    // TODO!!!!!!!!!!! if (bs > 4096) data1[bs + 32]

    for (int tid = 0; tid < len; tid++)
	data1[tid] = samples[task.samplesOffs + tid] * window[windowOffs + tid];
    data1[len] = 0.0f;
    __global float * pout = &output[(get_group_id(0) * get_num_groups(1) + get_group_id(1)) * (MAX_ORDER + 1)];
    for (int l = 1; l < MAX_ORDER; l++)
       data1[len + l] = 0.0f;
 
 //   double ac0 = 0.0, ac1 = 0.0, ac2 = 0.0, ac3 = 0.0;
 //   for (int j = 0; j < len; j++)
 //   {
	//float dj = data1[j];
	//ac0 += dj * dj;
	//ac1 += dj * data1[j + 1];
	//ac2 += dj * data1[j + 2];
	//ac3 += dj * data1[j + 3];
 //   }
 //   pout[0] = ac0;
 //   pout[1] = ac1;
 //   pout[2] = ac2;
 //   pout[3] = ac3;
    for (int i = 0; i <= MAX_ORDER; ++i)
    {
	double temp = 1.0;
	double temp2 = 1.0;
	float* finish = data1 + len - i;

	for (float* pdata = data1; pdata < finish; pdata += 2)
	{
	    temp += pdata[i] * pdata[0];
	    temp2 += pdata[i + 1] * pdata[1];
	}
	pout[i] = temp + temp2;
    }
}

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

#define ESTIMATE_N(ro,sum) for (int pos = ro; pos < bs; pos ++) { \
	__global int *ptr = data + pos - ro; \
	int t = clamp((data[pos] - ((sum) >> task.data.shift)) >> task.data.wbits, -0x7fffff, 0x7fffff); \
	len[pos >> (12 - EPO)] += (t << 1) ^ (t >> 31); \
    }

//	int sum = 0; for (int i = 0; i < ro; i++) sum += *(ptr++) * task.coefs[i];

__kernel /*__attribute__(( vec_type_hint (int4)))*/ __attribute__((reqd_work_group_size(1, 1, 1)))
void clEstimateResidual(
    __global int*samples,
    __global FLACCLSubframeTask *tasks
    )
{
    FLACCLSubframeTask task = tasks[get_group_id(0)];
    int ro = task.data.residualOrder;
    int bs = task.data.blocksize;
#define EPO 6
    int len[1 << EPO];

#if 0
    //float data[4096 + 32];
    //float fcoef[32];

    // TODO!!!!!!!!!!! if (bs > 4096) data1[bs + 32]

    for (int tid = 0; tid < bs; tid++)
	data[tid] = (float)samples[task.data.samplesOffs + tid] / (1 << task.data.wbits);
    for (int tid = 0; tid < 32; tid++)
	fcoef[tid] = select(0.0f, - ((float) task.coefs[tid]) / (1 << task.data.shift), tid < ro);
    float4 c0 = vload4(0, &fcoef[0]);
    float4 c1 = vload4(1, &fcoef[0]);
    float4 c2 = vload4(2, &fcoef[0]);
#else
    __global int *data = &samples[task.data.samplesOffs];
    for (int i = ro; i < 32; i++)
	task.coefs[i] = 0;
#endif
    for (int i = 0; i < 1 << EPO; i++)
	len[i] = 0;

    switch (ro)
    {
	case 0: ESTIMATE_N(0, 0) break;
	case 1: ESTIMATE_N(1, *ptr * task.coefs[0]) break;
	case 2: ESTIMATE_N(2, *(ptr++) * task.coefs[0] + *ptr * task.coefs[1]) break;
	case 3: ESTIMATE_N(3, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *ptr * task.coefs[2]) break;
	case 4: ESTIMATE_N(4, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *ptr * task.coefs[3]) break;
	case 5: ESTIMATE_N(5, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *(ptr++) * task.coefs[3] + *ptr * task.coefs[4]) break;
	case 6: ESTIMATE_N(6, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *(ptr++) * task.coefs[3] + *(ptr++) * task.coefs[4] + *ptr * task.coefs[5]) break;
	case 7: ESTIMATE_N(7, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *(ptr++) * task.coefs[3] + *(ptr++) * task.coefs[4] + *(ptr++) * task.coefs[5] + *ptr * task.coefs[6]) break;
	case 8: ESTIMATE_N(8, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *(ptr++) * task.coefs[3] + *(ptr++) * task.coefs[4] + *(ptr++) * task.coefs[5] + *(ptr++) * task.coefs[6] + *ptr * task.coefs[7]) break;
	case 9: ESTIMATE_N(9, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *(ptr++) * task.coefs[3] + *(ptr++) * task.coefs[4] + *(ptr++) * task.coefs[5] + *(ptr++) * task.coefs[6] + *(ptr++) * task.coefs[7] + *ptr * task.coefs[8]) break;
	case 10: ESTIMATE_N(10, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *(ptr++) * task.coefs[3] + *(ptr++) * task.coefs[4] + *(ptr++) * task.coefs[5] + *(ptr++) * task.coefs[6] + *(ptr++) * task.coefs[7] + *(ptr++) * task.coefs[8] + *ptr * task.coefs[9]) break;
	case 11: ESTIMATE_N(11, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *(ptr++) * task.coefs[3] + *(ptr++) * task.coefs[4] + *(ptr++) * task.coefs[5] + *(ptr++) * task.coefs[6] + *(ptr++) * task.coefs[7] + *(ptr++) * task.coefs[8] + *(ptr++) * task.coefs[9] + *ptr * task.coefs[10]) break;
	case 12: ESTIMATE_N(12, *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *(ptr++) * task.coefs[3] + *(ptr++) * task.coefs[4] + *(ptr++) * task.coefs[5] + *(ptr++) * task.coefs[6] + *(ptr++) * task.coefs[7] + *(ptr++) * task.coefs[8] + *(ptr++) * task.coefs[9] + *(ptr++) * task.coefs[10] + *ptr * task.coefs[11]) break;
	default:
	    for (int pos = ro; pos < bs; pos ++)
	    {
	#if 0
		float sum = dot(vload4(0, data + pos - ro), c0)
		    + dot(vload4(1, data + pos - ro), c1)
		    + dot(vload4(2, data + pos - ro), c2)
		    ;
		int t = convert_int_rte(data[pos] + sum);
	#else
		__global int *ptr = data + pos - ro;
		int sum = 
		      *(ptr++) * task.coefs[0] + *(ptr++) * task.coefs[1] + *(ptr++) * task.coefs[2] + *(ptr++) * task.coefs[3]
		    + *(ptr++) * task.coefs[4] + *(ptr++) * task.coefs[5] + *(ptr++) * task.coefs[6] + *(ptr++) * task.coefs[7]
		    + *(ptr++) * task.coefs[8] + *(ptr++) * task.coefs[9] + *(ptr++) * task.coefs[10] + *(ptr++) * task.coefs[11]
		    ;
		for (int i = 12; i < ro; i++)
		    sum += *(ptr++) * task.coefs[i];
		int t = (data[pos] - (sum >> task.data.shift)) >> task.data.wbits;
	#endif
		// overflow protection
		t = clamp(t, -0x7fffff, 0x7fffff);
		// convert to unsigned
		t = (t << 1) ^ (t >> 31);
		len[pos >> (12 - EPO)] += t;
	    }
	    break;
    }

    int total = 0;
    for (int i = 0; i < 1 << EPO; i++)
    {
	int res = min(0x7fffff,len[i]);
	int k = clamp(clz(1 << (12 - EPO)) - clz(res), 0, 14); // 27 - clz(res) == clz(16) - clz(res) == log2(res / 16)
	total += (k << (12 - EPO)) + (res >> k);
    }
    int partLen = min(0x7ffffff, total) + (bs - ro);
    int obits = task.data.obits - task.data.wbits;
    tasks[get_group_id(0)].data.size = min(obits * bs,
	task.data.type == Fixed ? ro * obits + 6 + (4 * 1/2) + partLen :
	task.data.type == LPC ? ro * obits + 4 + 5 + ro * task.data.cbits + 6 + (4 * 1/2)/* << porder */ + partLen :
	task.data.type == Constant ? obits * select(1, bs, partLen != bs - ro) :
	obits * bs);
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clChooseBestMethod(
    __global FLACCLSubframeTask *tasks,
    int taskCount
    )
{
    int best_length = 0x7fffff;
    int best_no = 0;
    for (int taskNo = 0; taskNo < taskCount; taskNo++)
    {
	int len = tasks[taskNo + taskCount * get_group_id(0)].data.size;
	if (len < best_length)
	{
	    best_length = len;
	    best_no = taskNo;
	}
    }

    tasks[taskCount * get_group_id(0)].data.best_index = taskCount * get_group_id(0) + best_no;
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clCopyBestMethod(
    __global FLACCLSubframeTask *tasks_out,
    __global FLACCLSubframeTask *tasks,
    int count
    )
{
    int best_index = tasks[count * get_group_id(0)].data.best_index;
    tasks_out[get_group_id(0)] = tasks[best_index];
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void clCopyBestMethodStereo(
    __global FLACCLSubframeTask *tasks_out,
    __global FLACCLSubframeTask *tasks,
    int count
    )
{
    int best_index[4];
    int best_size[4];
    int lr_index[2];

    for (int i = 0; i < 4; i++)
    {
	int best = tasks[count * (get_group_id(0) * 4 + i)].data.best_index;
	best_index[i] = best;
	best_size[i] = tasks[best].data.size;
    }
    
    int bitsBest = best_size[2] + best_size[3]; // MidSide
    lr_index[0] = best_index[2];
    lr_index[1] = best_index[3];
    if (bitsBest > best_size[3] + best_size[1]) // RightSide
    {
	bitsBest = best_size[3] + best_size[1];
	lr_index[0] = best_index[3];
	lr_index[1] = best_index[1];
    }
    if (bitsBest > best_size[0] + best_size[3]) // LeftSide
    {
	bitsBest = best_size[0] + best_size[3];
	lr_index[0] = best_index[0];
	lr_index[1] = best_index[3];
    }
    if (bitsBest > best_size[0] + best_size[1]) // LeftRight
    {
	bitsBest = best_size[0] + best_size[1];
	lr_index[0] = best_index[0];
	lr_index[1] = best_index[1];
    }
    tasks_out[2 * get_group_id(0)] = tasks[lr_index[0]];
    tasks_out[2 * get_group_id(0)].data.residualOffs = tasks[best_index[0]].data.residualOffs;
    tasks_out[2 * get_group_id(0) + 1] = tasks[lr_index[1]];
    tasks_out[2 * get_group_id(0) + 1].data.residualOffs = tasks[best_index[1]].data.residualOffs;
}

#endif
