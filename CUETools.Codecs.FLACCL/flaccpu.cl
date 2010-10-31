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

#define TEMPBLOCK 64

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
    double ac[MAX_ORDER + 4];
    
    for (int i = 0; i <= MAX_ORDER; ++i)
	ac[i] = 0.0;

    for (int pos = 0; pos < len; pos += TEMPBLOCK)
    {
	for (int tid = 0; tid < TEMPBLOCK + MAX_ORDER + 3; tid++)
	    data[tid] = tid < len - pos ? samples[task.samplesOffs + pos + tid] * window[windowOffs + pos + tid] : 0.0f;
 
	for (int i = 0; i <= MAX_ORDER; i += 4)
	{
	    float4 temp = 0.0;
	    for (int j = 0; j < min(TEMPBLOCK, len - pos); j++)
		temp += data[j] * vload4(0, &data[j + i]);
	    ac[i] += temp.x;
	    ac[i+1] += temp.y;
	    ac[i+2] += temp.z;
	    ac[i+3] += temp.w;
	}
    }
    __global float * pout = &output[(get_group_id(0) * get_num_groups(1) + get_group_id(1)) * (MAX_ORDER + 1)];
    for (int i = 0; i <= MAX_ORDER; ++i)
	pout[i] = ac[i];
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
    __global FLACCLSubframeTask *tasks
    )
{
    FLACCLSubframeTask task = tasks[get_group_id(0)];
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
    FLACCLSubframeTask task = tasks[get_group_id(0)];
    int bs = task.data.blocksize;
    int ro = task.data.residualOrder;
    __global int *data = &samples[task.data.samplesOffs];
    __global int *pl = partition_lengths + (1 << (max_porder + 1)) * get_group_id(0);
    for (int p = 0; p < (1 << max_porder); p++)
	pl[p] = 0;
    SWITCH_N((residual[task.data.residualOffs + pos] = t, t = clamp(t, -0x7fffff, 0x7fffff), t = (t << 1) ^ (t >> 31), pl[pos >> 4] += t));
}

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
    int psize = task->data.blocksize >> max_porder;
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
    task->data.size =
	task->data.type == Fixed ? task->data.residualOrder * obits + 6 + best_length :
	task->data.type == LPC ? task->data.residualOrder * obits + 6 + best_length + 4 + 5 + task->data.residualOrder * task->data.cbits :
	task->data.type == Constant ? obits : obits * task->data.blocksize;
    for (int offs = 0; offs < (1 << best_porder); offs ++)
	best_rice_parameters[(get_group_id(0) << max_porder) + offs] = rice_parameters[pos - (2 << best_porder) + offs];
}
#endif
