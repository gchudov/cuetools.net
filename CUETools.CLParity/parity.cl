/**
 * CUETools.CLParity: Reed-Solomon (32 bit) using OpenCL
 * Copyright (c) 2010-2022 Gregory S. Chudov
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

#define WARP_SIZE 32

int
galois_mul(int x, int y)
{
    const int poly = 00020000007;
    int prod = 0;

//#pragma unroll 0
    for (int i = 0; i < 4; i++)
    {
	if (x & 1) prod ^= y;
	y = (y << 1) ^ ((y >> 31) ? poly : 0);
	if (x & 2) prod ^= y;
	y = (y << 1) ^ ((y >> 31) ? poly : 0);
	if (x & 4) prod ^= y;
	y = (y << 1) ^ ((y >> 31) ? poly : 0);
	if (x & 8) prod ^= y;
	y = (y << 1) ^ ((y >> 31) ? poly : 0);
	if (x & 16) prod ^= y;
	y = (y << 1) ^ ((y >> 31) ? poly : 0);
	if (x & 32) prod ^= y;
	y = (y << 1) ^ ((y >> 31) ? poly : 0);
	if (x & 64) prod ^= y;
	y = (y << 1) ^ ((y >> 31) ? poly : 0);
	if (x & 128) prod ^= y;
	y = (y << 1) ^ ((y >> 31) ? poly : 0);
	x >>= 8;
    }
    return prod;
}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void reedSolomonInit(__global int *encodeGx0, __global int *encodeGx1, __global int *parity0, __global int *parity1, int npar)
{
    int i = get_group_id(0) * GROUP_SIZE + get_local_id(0);
    parity0[i] = parity1[i] = 0;
    encodeGx0[i] = encodeGx1[i] = (i == npar - 1 ? 1 : 0);
    if (i < GROUP_SIZE) encodeGx0[npar + i] = encodeGx1[npar + i] = 0;
    if (i == 0) parity0[npar] = parity1[npar] = 0;
}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void reedSolomonInitGx(__global int *exp, __global int *old_encodeGx, __global int *new_encodeGx, int npar, int step)
{
    int tid = get_local_id(0);
    int i = get_group_id(0) * (GROUP_SIZE / 2) + tid;
    __local int gx[GROUP_SIZE];
    __local int ex[GROUP_SIZE / 2];

    gx[tid] = old_encodeGx[i];
    if (tid < GROUP_SIZE / 2) ex[tid] = exp[step * (GROUP_SIZE / 2) + tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 0; s < GROUP_SIZE / 2; s++)
    {
	int p = tid + s < (GROUP_SIZE - 1) ? galois_mul(gx[tid], ex[s]) ^ gx[tid + 1] : 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	gx[tid] = p;
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < GROUP_SIZE / 2) new_encodeGx[i] = gx[tid];
}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void reedSolomonA(__global int* data, __global int*encodeGx, __global int*old_parity, int npar, int offset)
{
    int tid = get_local_id(0);
    __local int ib[GROUP_SIZE];
    __local int gx[GROUP_SIZE];

    ib[tid] = data[offset + tid] ^ old_parity[tid];
    gx[tid] = encodeGx[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 0; s < GROUP_SIZE; s++)
    {
	if (tid > s)
	    ib[tid] ^= galois_mul(ib[s], gx[tid - 1 - s]);
	barrier(CLK_LOCAL_MEM_FENCE);
    }

    data[offset + tid] = ib[tid];
}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void reedSolomonB(__global int* data, __global int* encodeGx, __global int* parity, int npar, int offset)
{
    __local int ib[GROUP_SIZE];
    __local int gx[GROUP_SIZE * 2];
    int tid = get_local_id(0);
    int i = (get_group_id(0) * GROUP_SIZE) + get_local_id(0);
    ib[tid] = data[offset + tid];
    gx[tid] = encodeGx[i];
    gx[tid + GROUP_SIZE] = i + GROUP_SIZE >= npar ? 0 : encodeGx[i + GROUP_SIZE];
    barrier(CLK_LOCAL_MEM_FENCE);

    int p = i + GROUP_SIZE >= npar ? 0 : parity[i + GROUP_SIZE];
    for (int s = 0; s < GROUP_SIZE; s++)
        p ^= galois_mul(ib[s], gx[tid + GROUP_SIZE - 1 - s]);
    parity[i] = p;
}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void reedSolomon(__global int*data, __global int*encodeGx, __global int*old_parity, __global int *new_parity, int npar, int offset)
{
    int tid = get_local_id(0);
    int i = (get_group_id(0) * GROUP_SIZE) + get_local_id(0);
    __local int old[GROUP_SIZE];
    __local int ib;
    if (tid == 0)
	ib = old_parity[0] ^ data[offset];
    if (tid == 0)
	old[GROUP_SIZE - 1] = old_parity[i + GROUP_SIZE];
    else
	old[tid - 1] = old_parity[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    new_parity[i] = old[tid] ^ galois_mul(ib, encodeGx[i]);
    //new_parity[i] = old_parity[i + 1] ^ galois_mul(ib, encodeGx[i]);
}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void reedSolomonDecodeInit(__global int* data, int dataLen, __global int*expTbl, __global int*syn, int npar)
{
    int i = (get_group_id(0) * GROUP_SIZE) + get_local_id(0);
    syn[i] = 0;
}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void reedSolomonDecode(__global int* data, int dataLen, __global int*expTbl, __global int*syn, int npar)
{
    int tid = get_local_id(0);
    int i = (get_group_id(0) * GROUP_SIZE) + get_local_id(0);
    __local int ds[GROUP_SIZE];
    
    ds[tid] = data[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    int wk = syn[i];
    int ai = expTbl[i];

    for (int s = 0; s < GROUP_SIZE; s++)
	wk = ds[s] ^ galois_mul(wk, ai);

    syn[i] = wk;
}

int galois_exp(int t, int n)
{
    int e = 1;
    for (int i = 0; i < 32; i++)
    {
	if ((n >> i) & 1) e = galois_mul(e, t);
	t = galois_mul(t, t);
    }
    return e;
}

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void chienSearch(__global int *wk_out, int pos, __global int *sigma, int jisu, int n)
{
    const int poly = 00020000007;
    const int oneByTwo = (poly >> 1) | (1 << 31);
    int tid = get_local_id(0);
    int i = (get_group_id(0) * GROUP_SIZE) + get_local_id(0);
    int zexp = galois_exp(oneByTwo, i + pos);
    int wk = 1;
    int jzexp = 1;
    __local int sj;

    for (int j = 1; j <= jisu; j++)
    {
	jzexp = galois_mul(jzexp, zexp);
	if (tid == 0) sj = sigma[j];
	barrier(CLK_LOCAL_MEM_FENCE);
	wk ^= galois_mul(sj, jzexp); // wk ^= sigma[j] / 2^(ij);
	barrier(CLK_LOCAL_MEM_FENCE);
    }
    wk_out[i] = wk;
}
