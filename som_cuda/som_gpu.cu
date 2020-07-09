/*
 * --------------------------------------------------------------------------- *
 *                                  CAMPAIGN                                   *
 * --------------------------------------------------------------------------- *
 * This is part of the CAMPAIGN data clustering library originating from       *
 * Simbios, the NIH National Center for Physics-Based Simulation of Biological *
 * Structures at Stanford, funded under the NIH Roadmap for Medical Research,  *
 * grant U54 GM072970 (See https://simtk.org), and the FEATURE Project at      *
 * Stanford, funded under the NIH grant LM05652                                *
 * (See http://feature.stanford.edu/index.php).                                *
 *                                                                             *
 * Portions copyright (c) 2010 Stanford University, Authors, and Contributors. *
 * Authors: Kai J. Kolhoff                                                     *
 * Contributors: Marc Sosnick, William Hsu                                     *
 *                                                                             *
 * This program is free software: you can redistribute it and/or modify it     *
 * under the terms of the GNU Lesser General Public License as published by    *
 * the Free Software Foundation, either version 3 of the License, or (at your  *
 * option) any later version.                                                  *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public        *
 * License for more details.                                                   *
 *                                                                             *
 * You should have received a copy of the GNU Lesser General Public License    *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.       *
 * --------------------------------------------------------------------------- *
 */

/* $Id: somGPU.cu 158 2011-01-12 23:06:27Z msosnick $ */
/**
 * \File somGPU.cu
 * \brief A CUDA self organizing map implementation
 *
 * Implements self-organizing map (som) clustering on the GPU
 * 
 * \author Author: Kai Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 */

#include "som_gpu.h"

using namespace std;


template <unsigned int BLOCKSIZE, class T, class U>
__device__ static void reduceMinTwo(int tid, T *s_A, U *s_B)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) 
    { 
        // first line assures same sequence as sequential code; removing this feature can improve efficiency
        if (s_A[tid + 512] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 512]);
        if (s_A[tid + 512] <  s_A[tid]) { s_A[tid] = s_A[tid + 512]; s_B[tid] = s_B[tid + 512]; } 
    } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) 
    { 
        if (s_A[tid + 256] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 256]);
        if (s_A[tid + 256] <  s_A[tid]) { s_A[tid] = s_A[tid + 256]; s_B[tid] = s_B[tid + 256]; } 
    } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) 
    {
        if (s_A[tid + 128] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 128]);
        if (s_A[tid + 128] <  s_A[tid]) { s_A[tid] = s_A[tid + 128]; s_B[tid] = s_B[tid + 128]; } 
    } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) 
    { 
        if (s_A[tid +  64] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  64]);
        if (s_A[tid +  64] <  s_A[tid]) { s_A[tid] = s_A[tid +  64]; s_B[tid] = s_B[tid +  64]; } 
    } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { if (s_A[tid + 32] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 32]); if (s_A[tid + 32] < s_A[tid]) { s_A[tid] = s_A[tid + 32]; s_B[tid] = s_B[tid + 32]; } }
        if (BLOCKSIZE >= 32) { if (s_A[tid + 16] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 16]); if (s_A[tid + 16] < s_A[tid]) { s_A[tid] = s_A[tid + 16]; s_B[tid] = s_B[tid + 16]; } }
        if (BLOCKSIZE >= 16) { if (s_A[tid +  8] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  8]); if (s_A[tid +  8] < s_A[tid]) { s_A[tid] = s_A[tid +  8]; s_B[tid] = s_B[tid +  8]; } }
        if (BLOCKSIZE >=  8) { if (s_A[tid +  4] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  4]); if (s_A[tid +  4] < s_A[tid]) { s_A[tid] = s_A[tid +  4]; s_B[tid] = s_B[tid +  4]; } }
        if (BLOCKSIZE >=  4) { if (s_A[tid +  2] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  2]); if (s_A[tid +  2] < s_A[tid]) { s_A[tid] = s_A[tid +  2]; s_B[tid] = s_B[tid +  2]; } }
        if (BLOCKSIZE >=  2) { if (s_A[tid +  1] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  1]); if (s_A[tid +  1] < s_A[tid]) { s_A[tid] = s_A[tid +  1]; s_B[tid] = s_B[tid +  1]; } }
    }
}



__global__ static void findBMU_CUDA(int N, int K, int D, int v, FLOAT_TYPE *X, FLOAT_TYPE *WV, int *BMU, FLOAT_TYPE *DISTS)
{
    extern __shared__ FLOAT_TYPE array[];                   // declares variable in shared memory
    FLOAT_TYPE *s_data   = (FLOAT_TYPE*) array;                  // dynamically allocate array to hold position data for tpb FLOAT_TYPEs
    FLOAT_TYPE *s_dist   = (FLOAT_TYPE*) &s_data[blockDim.x];    // dynamically allocate array to hold intermediate distance results
    FLOAT_TYPE *s_inputV = (FLOAT_TYPE*) &s_dist[blockDim.x];    // dynamically allocate variable to hold components of input vector
    
    int t   = blockDim.x * blockIdx.x + threadIdx.x;   // global thread ID
    int tid = threadIdx.x;                             // in-block thread ID
    
    s_dist[tid] = FLT_MAX;
    if (t < K) s_dist[tid] = 0.0;
    
    // for each dimension
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb components of input vector to shared memory (non-coalesced)
        if (offsetD + tid < D) s_inputV[tid] = X[(offsetD + tid) * N + v];
        __syncthreads();
        // compute distances between up to tpb weight vectors and input vector components
        for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
        {
            if (t < K) s_dist[tid] += distanceComponentGPU(WV + d * K + t, s_inputV + d - offsetD);
        }
        offsetD += blockDim.x;
        __syncthreads();
    }
    s_dist[tid] = distanceFinalizeGPU(1, s_dist + tid); 
    
    // now s_dist contains blockDim.x distances; reduce to find best matching unit in block     
    
    // reuse s_data
    s_data[tid] = tid;
    __syncthreads();
    
    // reduce sdata and iresult, to get minimum distance and bmu index
    reduceMinTwo<THREADSPERBLOCK>(tid, s_dist, s_data);
    
    // return values
    if (tid == 0)
    {
        BMU[blockIdx.x]   = (int) s_data[tid];
        DISTS[blockIdx.x] = s_dist[tid];
    }      
}



__global__ static void copyBMU_CUDA(int K, int D, int bmu, FLOAT_TYPE *WV, FLOAT_TYPE *BMU)
{
    int tid = threadIdx.x;
    if (tid < D)
    {
        // not coalesced
        BMU[tid] = WV[tid * K + bmu];
    }
}


__global__ static void updateNeighborhood_CUDA(int N, int K, int D, int v, FLOAT_TYPE d0, FLOAT_TYPE scale, FLOAT_TYPE *BMU, FLOAT_TYPE *X, FLOAT_TYPE *WV)
{
    extern __shared__ FLOAT_TYPE array[];                       // declares variable in shared memory
    FLOAT_TYPE *s_bmuDist    = (FLOAT_TYPE*) array;                  // dynamically allocate array to hold intermediate distance results
    FLOAT_TYPE *s_vectorComp = (FLOAT_TYPE*) &s_bmuDist[blockDim.x]; // dynamically allocate variable to hold components of input vector
    
    int t   = blockDim.x * blockIdx.x + threadIdx.x;       // global thread ID
    int tid = threadIdx.x;                                 // in-block thread ID
    
    s_bmuDist[tid] = 0.0;
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb bmu vector components to shared memory (coalesced)
        if (offsetD + tid < D) s_vectorComp[tid] = BMU[offsetD + tid];
        __syncthreads();
        // compute distances for up to tpb dimensions
        for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
        {
            if (t < K) s_bmuDist[tid] += distanceComponentGPU(WV + d * K + t, s_vectorComp + d - offsetD);
        }
        offsetD += blockDim.x;
        __syncthreads();
    }
    if (t < K) s_bmuDist[tid] = distanceFinalizeGPU(1, s_bmuDist + tid);
    __syncthreads();
    // now update weight vector position towards inputV using d0, bmuDist, and learning restraint
    offsetD = 0;
    while (offsetD < D)
    {
        // read up to tpb components from input vector (non-coalesced)
        if (offsetD + tid < D) s_vectorComp[tid] = X[(offsetD + tid) * N + v];
        __syncthreads();
        // modify up to tpb components of up to tpb weight vectors (coalesced)
        for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
        {
            if (t < K) WV[d * K + t] += scale * pow((FLOAT_TYPE) 0.25, s_bmuDist[tid] / d0) * (s_vectorComp[d - offsetD] - WV[d * K + t]);
        }
        offsetD += blockDim.x;
        __syncthreads();
    }
}



void somGPU(int N, int K, int D, int numIter, FLOAT_TYPE *x, FLOAT_TYPE **pwv, DataIO *data)
{
    FLOAT_TYPE *wv = *pwv;
    
    // determine CUDA parameters
    dim3 block(THREADSPERBLOCK);        
    dim3 gridK2((int) ceil((FLOAT_TYPE) K / (FLOAT_TYPE) THREADSPERBLOCK)); 
    int numBlocks = (int) ceil((FLOAT_TYPE) K / (FLOAT_TYPE) THREADSPERBLOCK);
    dim3 gridK(numBlocks);
    dim3 gridD((int) ceil((FLOAT_TYPE) D / (FLOAT_TYPE) THREADSPERBLOCK));
    int sMemBMU = sizeof(FLOAT_TYPE) * 3 * THREADSPERBLOCK; // for BMU search kernel
    int sMemNei = sizeof(FLOAT_TYPE) * 2 * THREADSPERBLOCK; // for neighborhood update kernel
    
    // GPU memory pointers, allocate and initialize memory
    void * x_d      = data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N * D, x);
    void * wv_d     = data->allocDeviceMemory(sizeof(FLOAT_TYPE) * K * D, wv);
    void * bmuIDs_d = data->allocDeviceMemory  (sizeof(int) * numBlocks);
    void * bmu_d    = data->allocDeviceMemory(sizeof(FLOAT_TYPE) * D);
    void * dists_d  = data->allocDeviceMemory(sizeof(FLOAT_TYPE) * numBlocks);
    
    int   *bmuIDs   = (int*)   malloc(sizeof(int)   * numBlocks);
    FLOAT_TYPE *dists    = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * numBlocks);
    
    // for each iteration
    for (unsigned int iter = 0; iter < numIter; iter++)
    {
        // learning restraint, between 1.0 and 0.0 and continuously decreasing
        FLOAT_TYPE learningRestraint = 1.0f - (FLOAT_TYPE) iter / (FLOAT_TYPE) numIter;
        // for each input vector
        for (unsigned int v = 0; v < N; v++)
        {
            // find best matching unit (bmu) among weight vectors
            findBMU_CUDA<<<gridK, block, sMemBMU>>>(N, K, D, v, (FLOAT_TYPE*)x_d, (FLOAT_TYPE*)wv_d, (int*)bmuIDs_d, (FLOAT_TYPE*)dists_d);
            //CUT_CHECK_ERROR("findBMU_CUDA() kernel execution failed");

            // finish reduction on CPU
            cudaMemcpy(bmuIDs, (int*)bmuIDs_d, sizeof(int)   * numBlocks, cudaMemcpyDeviceToHost);
            cudaMemcpy(dists, (FLOAT_TYPE*)dists_d,  sizeof(FLOAT_TYPE) * numBlocks, cudaMemcpyDeviceToHost);
            int bmuID = bmuIDs[0];
            FLOAT_TYPE minDist = dists[0];
            for (unsigned int i = 1; i < numBlocks; i++)
            {
                if (dists[i] < minDist)
                {
                    minDist = dists[i];
                    bmuID   = bmuIDs[i];
                }
            }
            // got bmu and dist(bmu, inputV), make a copy of bmu
            copyBMU_CUDA<<<gridD, block>>>(K, D, bmuID, (FLOAT_TYPE*)wv_d, (FLOAT_TYPE*)bmu_d);
            //CUT_CHECK_ERROR("copyBMU_CUDA() kernel execution failed");
            
            // moves wv towards input vector
            updateNeighborhood_CUDA<<<gridK, block, sMemNei>>>(N, K, D, v, minDist, learningRestraint, (FLOAT_TYPE*)bmu_d, (FLOAT_TYPE*)x_d, (FLOAT_TYPE*)wv_d);
            //CUT_CHECK_ERROR("updateNeighborhood_CUDA() kernel execution failed");
        }
    }
    
    // copy weight vector values back to the host
    cudaMemcpy(wv, wv_d, sizeof(FLOAT_TYPE) * K * D, cudaMemcpyDeviceToHost); 
    
    // free memory
    cudaFree(bmu_d);
    cudaFree(wv_d);
    cudaFree(x_d);
    
    *pwv = wv;
}

