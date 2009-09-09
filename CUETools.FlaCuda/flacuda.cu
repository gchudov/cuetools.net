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

extern "C" __global__ void cudaComputeAutocor(
    float *output,
    const int *samples,
    const float *window,
    computeAutocorTaskStruct *tasks,
    int max_order, // should be <= 32
    int frameSize,
    int partSize // should be <= blockDim - max_order
)
{
    __shared__ struct {
	float data[256];
	float product[256];
	float product2[256];
	float sum[33];
	computeAutocorTaskStruct task;
    } shared;
    const int tid = threadIdx.x;
    // fetch task data
    if (tid < sizeof(shared.task) / sizeof(int))
	((int*)&shared.task)[tid] = ((int*)(tasks + blockIdx.y))[tid];
    __syncthreads();

    const int pos = blockIdx.x * partSize;
    const int productLen = min(frameSize - pos - max_order, partSize);
    const int dataLen = productLen + max_order;

    // fetch samples
    shared.data[tid] = tid < dataLen ? samples[shared.task.samplesOffs + pos + tid] * window[shared.task.windowOffs + pos + tid]: 0.0f;
    __syncthreads();

    for (int lag = 0; lag <= max_order; lag++)
    {
	shared.product[tid] = tid < productLen ? shared.data[tid] * shared.data[tid + lag] : 0.0f;
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
    }

    // return results
    if (tid <= max_order)
	output[(blockIdx.x + blockIdx.y * gridDim.x) * (max_order + 1) + tid] = shared.sum[tid];
}

typedef struct
{
    int residualOrder; // <= 32
    int samplesOffs;
    int shift;
    int reserved;
    int coefs[32];
} encodeResidualTaskStruct;

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
    const int residualOrder = shared.task.residualOrder;
    const int residualLen = min(frameSize - pos - residualOrder - 1, partSize);
    const int dataLen = residualLen + residualOrder + 1;

    // fetch samples
    shared.data[tid] = (tid < dataLen ? samples[shared.task.samplesOffs + pos + tid] : 0);
    
    // compute residual
    __syncthreads();
    long sum = 0;
    for (int c = 0; c <= residualOrder; c++)
	sum += __mul24(shared.data[tid + c], shared.task.coefs[residualOrder - c]);
    int res = shared.data[tid + residualOrder + 1] - (sum >> shared.task.shift);
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
