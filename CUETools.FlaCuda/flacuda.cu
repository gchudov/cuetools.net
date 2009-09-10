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
} computeAutocorTaskStruct;

typedef struct
{
    int residualOrder; // <= 32
    int samplesOffs;
    int shift;
    int cbits;
    int size;
    int reserved[11];
    int coefs[32];
} encodeResidualTaskStruct;

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
	float product[256];
	float sum[33];
	computeAutocorTaskStruct task;
    } shared;
    const int tid = threadIdx.x;
    const int tid2 = threadIdx.x + 256;
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    __syncthreads();

    const int pos = blockIdx.x * partSize;
    const int productLen = min(frameSize - pos - max_order, partSize);
    const int dataLen = productLen + max_order;

    // fetch samples
    shared.data[tid] = tid < dataLen ? samples[shared.task.samplesOffs + pos + tid] * window[shared.task.windowOffs + pos + tid]: 0.0f;
    shared.data[tid2] = tid2 < dataLen ? samples[shared.task.samplesOffs + pos + tid2] * window[shared.task.windowOffs + pos + tid2]: 0.0f;
    __syncthreads();

    for (int lag = 0; lag <= max_order; lag++)
    {
	shared.product[tid] = (tid < productLen) * shared.data[tid] * shared.data[tid + lag] +
	    + (tid2 < productLen) * shared.data[tid2] * shared.data[tid2 + lag];
	__syncthreads();

	// product sum: reduction in shared mem
	//if (tid < 256) shared.product[tid] += shared.product[tid + 256]; __syncthreads();
	if (tid < 128) shared.product[tid] += shared.product[tid + 128]; __syncthreads();
	if (tid < 64) shared.product[tid] += shared.product[tid + 64]; __syncthreads();
	if (tid < 32) shared.product[tid] += shared.product[tid + 32]; __syncthreads();
	shared.product[tid] += shared.product[tid + 16];
	shared.product[tid] += shared.product[tid + 8];
	shared.product[tid] += shared.product[tid + 4];
	shared.product[tid] += shared.product[tid + 2];
	if (tid == 0) shared.sum[lag] = shared.product[0] + shared.product[1]; 
	__syncthreads();
    }

    // return results
    if (tid <= max_order)
	output[(blockIdx.x + blockIdx.y * gridDim.x) * (max_order + 1) + tid] = shared.sum[tid];
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
	float tmp[32];
	float buf[32];
	int   bits[32];
	float autoc[33];
	int   cbits;
    } shared;
    const int tid = threadIdx.x;
    
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    
    // initialize autoc sums
    if (tid <= max_order) 
	shared.autoc[tid] = 0.0f;
    
    __syncthreads();

    // add up parts
    for (int part = 0; part < partCount; part++)
	if (tid <= max_order)
	    shared.autoc[tid] += autoc[(blockIdx.y * partCount + part) * (max_order + 1) + tid];
    
    __syncthreads();

    if (tid < 32)
	shared.tmp[tid] = 0.0f;

    float err = shared.autoc[0];

    for(int order = 0; order < max_order; order++)
    {
	if (tid < 32)
	{
	    shared.buf[tid] = (tid < order) * shared.tmp[tid] * shared.autoc[order - tid];
	    shared.buf[tid] += shared.buf[tid + 16];
	    shared.buf[tid] += shared.buf[tid + 8];
	    shared.buf[tid] += shared.buf[tid + 4];
	    shared.buf[tid] += shared.buf[tid + 2];
	    shared.buf[tid] += shared.buf[tid + 1];
	}
	__syncthreads();
        
	float r = (- shared.autoc[order+1] - shared.buf[0]) / err;
        
	err *= 1.0f - (r * r);
        
	shared.tmp[tid] += (tid < order) * r * shared.tmp[order - 1 - tid] + (tid  == order) * r;
	
	if (tid < 32)
	{
	    int precision = 13;
	    shared.bits[tid] = __mul24((33 - __clz(__float2int_rn(fabs(shared.tmp[tid]) * (1 << 15))) - precision), tid <= order);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 16]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 8]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 4]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 2]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 1]);
	    int sh = max(0,min(15, 15 - shared.bits[0]));
	    int coef = max(-(1 << precision),min((1 << precision)-1,__float2int_rn(-shared.tmp[tid] * (1 << sh))));
	    if (tid <= order)
		output[(blockIdx.x + blockIdx.y * gridDim.x) * max_order + order].coefs[tid] = coef;
	    if (tid == 0)
		output[(blockIdx.x + blockIdx.y * gridDim.x) * max_order + order].shift = sh;
	    shared.bits[tid] = 33 - max(__clz(coef),__clz(-1 ^ coef));
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 16]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 8]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 4]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 2]);
	    shared.bits[tid] = max(shared.bits[tid], shared.bits[tid + 1]);
	    int cbits = shared.bits[0];
	    if (tid == 0)
		output[(blockIdx.x + blockIdx.y * gridDim.x) * max_order + order].cbits = cbits;
	}
	__syncthreads();
    }
}

extern "C" __global__ void cudaEstimateResidual(
    int*output,
    int*samples,
    encodeResidualTaskStruct *tasks,
    int frameSize,
    int partSize // should be <= blockDim - max_order
    )
{
    __shared__ struct {
	int data[256];
	int residual[256];
	int rice[32];
	encodeResidualTaskStruct task;
    } shared;
    const int tid = threadIdx.x;
    // fetch task data
    if (tid < sizeof(encodeResidualTaskStruct) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    __syncthreads();
    const int pos = blockIdx.x * partSize;
    const int residualOrder = shared.task.residualOrder + 1;
    const int residualLen = min(frameSize - pos - residualOrder, partSize);
    const int dataLen = residualLen + residualOrder;

    // fetch samples
    shared.data[tid] = (tid < dataLen ? samples[shared.task.samplesOffs + pos + tid] : 0);

    // reverse coefs
    if (tid < residualOrder) shared.task.coefs[tid] = shared.task.coefs[residualOrder - 1 - tid];
    
    // compute residual
    __syncthreads();
    long sum = 0;
    for (int c = 0; c < residualOrder; c++)
	sum += __mul24(shared.data[tid + c], shared.task.coefs[c]);
    int res = shared.data[tid + residualOrder] - (sum >> shared.task.shift);
    shared.residual[tid] = __mul24(tid < residualLen, (2 * res) ^ (res >> 31));
    
    __syncthreads();
    // residual sum: reduction in shared mem
    if (tid < 128) shared.residual[tid] += shared.residual[tid + 128]; __syncthreads();
    if (tid < 64) shared.residual[tid] += shared.residual[tid + 64]; __syncthreads();
    if (tid < 32) shared.residual[tid] += shared.residual[tid + 32]; __syncthreads();
    shared.residual[tid] += shared.residual[tid + 16];
    shared.residual[tid] += shared.residual[tid + 8];
    shared.residual[tid] += shared.residual[tid + 4];
    shared.residual[tid] += shared.residual[tid + 2];
    shared.residual[tid] += shared.residual[tid + 1];
    __syncthreads();

    if (tid < 32)
    {
	// rice parameter search
	shared.rice[tid] = __mul24(tid >= 15, 0x7fffff) + residualLen * (tid + 1) + ((shared.residual[0] - (residualLen >> 1)) >> tid);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 8]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 4]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 2]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 1]);
    }
    if (tid == 0)
	output[blockIdx.x + blockIdx.y * gridDim.x] = shared.rice[0];
}

extern "C" __global__ void cudaSumResidual(
    encodeResidualTaskStruct *tasks,
    int *residual,
    int partCount // <= blockDim.y (64)
    )
{
    __shared__ struct {
	int partLen[64];
	//encodeResidualTaskStruct task;
    } shared;

    const int tid = threadIdx.x;
    // fetch task data
 //   if (tid < sizeof(encodeResidualTaskStruct) / sizeof(int))
	//((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
 //   __syncthreads();

    shared.partLen[tid] = (tid < partCount) ? residual[tid + partCount * blockIdx.y] : 0;

    __syncthreads();
    // length sum: reduction in shared mem
    if (tid < 32) shared.partLen[tid] += shared.partLen[tid + 32]; __syncthreads();
    shared.partLen[tid] += shared.partLen[tid + 16];
    shared.partLen[tid] += shared.partLen[tid + 8];
    shared.partLen[tid] += shared.partLen[tid + 4];
    shared.partLen[tid] += shared.partLen[tid + 2];
    shared.partLen[tid] += shared.partLen[tid + 1];
    __syncthreads();
    
    // FIXME: should process partition order here!!!

    // return sum
    if (tid == 0)
	tasks[blockIdx.y].size = shared.partLen[0]; 
}

extern "C" __global__ void cudaEncodeResidual(
    int*output,
    int*samples,
    encodeResidualTaskStruct *tasks,
    int frameSize,
    int partSize // should be <= blockDim - max_order
    )
{
    __shared__ struct {
	int data[256];
	int residual[256];
	int rice[32];
	encodeResidualTaskStruct task;
    } shared;
    const int tid = threadIdx.x;
    // fetch task data
    if (tid < sizeof(encodeResidualTaskStruct) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    __syncthreads();
    const int pos = blockIdx.x * partSize;
    const int residualOrder = shared.task.residualOrder + 1;
    const int residualLen = min(frameSize - pos - residualOrder, partSize);
    const int dataLen = residualLen + residualOrder;

    // fetch samples
    shared.data[tid] = (tid < dataLen ? samples[shared.task.samplesOffs + pos + tid] : 0);

    // reverse coefs
    if (tid < residualOrder) shared.task.coefs[tid] = shared.task.coefs[residualOrder - 1 - tid];
    
    // compute residual
    __syncthreads();
    long sum = 0;
    for (int c = 0; c < residualOrder; c++)
	sum += __mul24(shared.data[tid + c], shared.task.coefs[c]);
    int res = shared.data[tid + residualOrder] - (sum >> shared.task.shift);
    shared.residual[tid] = __mul24(tid < residualLen, (2 * res) ^ (res >> 31));
    
    __syncthreads();
    // residual sum: reduction in shared mem
    if (tid < 128) shared.residual[tid] += shared.residual[tid + 128]; __syncthreads();
    if (tid < 64) shared.residual[tid] += shared.residual[tid + 64]; __syncthreads();
    if (tid < 32) shared.residual[tid] += shared.residual[tid + 32]; __syncthreads();
    shared.residual[tid] += shared.residual[tid + 16];
    shared.residual[tid] += shared.residual[tid + 8];
    shared.residual[tid] += shared.residual[tid + 4];
    shared.residual[tid] += shared.residual[tid + 2];
    shared.residual[tid] += shared.residual[tid + 1];
    __syncthreads();

    if (tid < 32)
    {
	// rice parameter search
	shared.rice[tid] = __mul24(tid >= 15, 0x7fffff) + residualLen * (tid + 1) + ((shared.residual[0] - (residualLen >> 1)) >> tid);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 8]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 4]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 2]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 1]);
    }
    if (tid == 0)
	output[blockIdx.x + blockIdx.y * gridDim.x] = shared.rice[0];
}
#endif
