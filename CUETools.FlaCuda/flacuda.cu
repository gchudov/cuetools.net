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
	volatile float ldr[32];
	volatile int   bits[32];
	volatile float autoc[33];
	volatile float gen0[32];
	volatile float gen1[32];
	//volatile float reff[32];
	int   cbits;
    } shared;
    const int tid = threadIdx.x;
    
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    
    // initialize autoc sums
    if (tid <= max_order) 
	shared.autoc[tid] = 0.0f;
    
    // add up parts
    for (int part = 0; part < partCount; part++)
	if (tid <= max_order)
	    shared.autoc[tid] += autoc[(blockIdx.y * partCount + part) * (max_order + 1) + tid];
    
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
	    error += shared.gen1[0] * reff;
	    if (tid < max_order - 1 - order)
	    {
		float g1 = shared.gen1[tid + 1] + reff * shared.gen0[tid];
		float g0 = shared.gen1[tid + 1] * reff + shared.gen0[tid];
		shared.gen1[tid] = g1;
		shared.gen0[tid] = g0;
	    }

	    // Levinson-Durbin recursion
	    shared.ldr[tid] += (tid < order) * reff * shared.ldr[order - 1 - tid] + (tid  == order) * reff;

	    // Quantization
	    int precision = 13;
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
	    shared.bits[tid] = 33 - max(__clz(coef),__clz(-1 ^ coef));
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

// blockDim.x == 32
// blockDim.y == 8
extern "C" __global__ void cudaEstimateResidual(
    int*output,
    int*samples,
    encodeResidualTaskStruct *tasks,
    int max_order,
    int frameSize,
    int partSize // should be 224
    )
{
    __shared__ struct {
	int data[32*8];
	int residual[32*8];
	encodeResidualTaskStruct task[8];
    } shared;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (threadIdx.x < 16)
	((int*)&shared.task[threadIdx.y])[threadIdx.x] = ((int*)(&tasks[blockIdx.y * blockDim.y + threadIdx.y]))[threadIdx.x];
    __syncthreads();
    const int pos = blockIdx.x * partSize;
    const int dataLen = min(frameSize - pos, partSize + max_order);

    // fetch samples
    shared.data[tid] = tid < dataLen ? samples[shared.task[0].samplesOffs + pos + tid] : 0;
    shared.residual[tid] = 0;
    const int residualLen = max(0,min(frameSize - pos - shared.task[threadIdx.y].residualOrder, partSize)) * (shared.task[threadIdx.y].residualOrder != 0);

    __syncthreads();

    shared.task[threadIdx.y].coefs[threadIdx.x] = threadIdx.x < max_order ? tasks[blockIdx.y * blockDim.y + threadIdx.y].coefs[threadIdx.x] : 0;

    for (int i = threadIdx.x; i - threadIdx.x < residualLen; i += blockDim.x) // += 32
    {
	// compute residual
	int sum = 0;
	int c = 0;
	for (c = 0; c < shared.task[threadIdx.y].residualOrder; c++)
	    sum += __mul24(shared.data[i + c], shared.task[threadIdx.y].coefs[c]);
	sum = shared.data[i + c] - (sum >> shared.task[threadIdx.y].shift);
	shared.residual[tid] += __mul24(i < residualLen, (sum << 1) ^ (sum >> 31));
    }

    // enable this line when using blockDim.x == 64
    //__syncthreads(); if (threadIdx.x < 32) shared.residual[tid] += shared.residual[tid + 32]; __syncthreads();
    shared.residual[tid] += shared.residual[tid + 16];
    shared.residual[tid] += shared.residual[tid + 8];
    shared.residual[tid] += shared.residual[tid + 4];
    shared.residual[tid] += shared.residual[tid + 2];
    shared.residual[tid] += shared.residual[tid + 1];

    // rice parameter search
    shared.residual[tid] = __mul24(threadIdx.x >= 15, 0x7fffff) + residualLen * (threadIdx.x + 1) + ((shared.residual[threadIdx.y * blockDim.x] - (residualLen >> 1)) >> threadIdx.x);
    __syncthreads();
    shared.residual[tid] = min(shared.residual[tid], shared.residual[tid + 8]);
    shared.residual[tid] = min(shared.residual[tid], shared.residual[tid + 4]);
    shared.residual[tid] = min(shared.residual[tid], shared.residual[tid + 2]);
    shared.residual[tid] = min(shared.residual[tid], shared.residual[tid + 1]);
    if (threadIdx.x == 0 && shared.task[threadIdx.y].residualOrder != 0)
	output[(blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x + blockIdx.x] = shared.residual[tid];
}

// blockDim.x == 256
// gridDim.x = frameSize / chunkSize
extern "C" __global__ void cudaSumResidualChunks(
    int *output,
    encodeResidualTaskStruct *tasks,
    int *residual,
    int frameSize,
    int chunkSize // <= blockDim.x(256)
    )
{
    __shared__ struct {
	int residual[256];
	int rice[32];
    } shared;
    
    // fetch parameters
    const int tid = threadIdx.x;
    const int residualOrder = tasks[blockIdx.y].residualOrder;
    const int chunkNumber = blockIdx.x;
    const int pos = chunkNumber * chunkSize;
    const int residualLen = min(frameSize - pos - residualOrder, chunkSize);

    // set upper residuals to zero, in case blockDim < 256
    shared.residual[255 - tid] = 0;

    // read residual
    int res = (tid < residualLen) ? residual[blockIdx.y * 8192 + pos + tid] : 0;

    // convert to unsigned
    shared.residual[tid] = (2 * res) ^ (res >> 31);
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

    if (tid < 32)
    {
	// rice parameter search
	shared.rice[tid] = __mul24(tid >= 15, 0x7fffff) + residualLen * (tid + 1) + ((shared.residual[0] - (residualLen >> 1)) >> tid);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 8]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 4]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 2]);
	shared.rice[tid] = min(shared.rice[tid], shared.rice[tid + 1]);
    }

    // write output
    if (tid == 0)
	output[blockIdx.x + blockIdx.y * gridDim.x] = shared.rice[0];
}

extern "C" __global__ void cudaSumResidual(
    encodeResidualTaskStruct *tasks,
    int *residual,
    int partSize,
    int partCount // <= blockDim.y (256)
    )
{
    __shared__ struct {
	int partLen[256];
	encodeResidualTaskStruct task;
    } shared;

    const int tid = threadIdx.x;
    // fetch task data
    if (tid < sizeof(encodeResidualTaskStruct) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    __syncthreads();

    shared.partLen[tid] = (tid < partCount) ? residual[tid + partCount * blockIdx.y] : 0;
    __syncthreads();
    // length sum: reduction in shared mem
    //if (tid < 128) shared.partLen[tid] += shared.partLen[tid + 128]; __syncthreads();
    //if (tid < 64) shared.partLen[tid] += shared.partLen[tid + 64]; __syncthreads();
    if (tid < 32) shared.partLen[tid] += shared.partLen[tid + 32]; __syncthreads();
    shared.partLen[tid] += shared.partLen[tid + 16];
    shared.partLen[tid] += shared.partLen[tid + 8];
    shared.partLen[tid] += shared.partLen[tid + 4];
    shared.partLen[tid] += shared.partLen[tid + 2];
    shared.partLen[tid] += shared.partLen[tid + 1];
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
    __syncthreads();
}
#endif
