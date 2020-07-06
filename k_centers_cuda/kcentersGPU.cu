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
 * Authors:  Kai J. Kohlhoff                                                   *
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

/* $Id$ */

/**
 * \file kcentersGPU.cu
 * \brief A CUDA K-centers implementation
 *
 * Implements parallel K-centers clustering on the GPU
 * 
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 12/2/2010
 * \version 1.0
 **/

#include "kcentersGPU.h"

using namespace std;


template <unsigned int BLOCKSIZE, class T, class U>
__device__ static void parallelMax(int tid, T *s_A, U *s_B)
{
    if (BLOCKSIZE >= 1024)
    {
      if (tid < 512 && s_A[tid + 512] > s_A[tid])
      {
        s_A[tid] = s_A[tid + 512];
        s_B[tid] = s_B[tid + 512];
      }
      __syncthreads();
    }
    if (BLOCKSIZE >=  512)
    {
      if (tid < 256 && s_A[tid + 256] > s_A[tid])
      {
        s_A[tid] = s_A[tid + 256];
        s_B[tid] = s_B[tid + 256];
      }
      __syncthreads();
    }
    if (BLOCKSIZE >=  256)
    {
      if (tid < 128 && s_A[tid + 128] > s_A[tid])
      {
        s_A[tid] = s_A[tid + 128];
        s_B[tid] = s_B[tid + 128];
      }
      __syncthreads();
    }
    if (BLOCKSIZE >=  128)
    {
      if (tid <  64 && s_A[tid +  64] > s_A[tid])
      {
        s_A[tid] = s_A[tid +  64];
        s_B[tid] = s_B[tid +  64];
      }
      __syncthreads();
    }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64)
        {
          if (s_A[tid + 32] > s_A[tid])
          {
            s_A[tid] = s_A[tid + 32];
            s_B[tid] = s_B[tid + 32];
          }
        }
        if (BLOCKSIZE >= 32)
        {
          if (s_A[tid + 16] > s_A[tid])
          {
            s_A[tid] = s_A[tid + 16];
            s_B[tid] = s_B[tid + 16];
          }
        }
        if (BLOCKSIZE >= 16)
        {
          if (s_A[tid +  8] > s_A[tid])
          {
            s_A[tid] = s_A[tid +  8];
            s_B[tid] = s_B[tid +  8];
          }
        }
        if (BLOCKSIZE >=  8)
        {
          if (s_A[tid +  4] > s_A[tid])
          {
            s_A[tid] = s_A[tid +  4];
            s_B[tid] = s_B[tid +  4];
          }
        }
        if (BLOCKSIZE >=  4)
        {
          if (s_A[tid +  2] > s_A[tid])
          {
            s_A[tid] = s_A[tid +  2];
            s_B[tid] = s_B[tid +  2];
          }
        }
        if (BLOCKSIZE >=  2)
        {
          if (s_A[tid +  1] > s_A[tid])
          {
            s_A[tid] = s_A[tid +  1];
            s_B[tid] = s_B[tid +  1];
          }
        }
    }
}



__global__ void checkCentroid_CUDA(int N, int D, int iter, int centroid, FLOAT_TYPE *X, FLOAT_TYPE *CTR, FLOAT_TYPE *DIST, int *ASSIGN, FLOAT_TYPE *MAXDIST, int *MAXID)
{
    extern __shared__ FLOAT_TYPE array[];                  // shared memory
    FLOAT_TYPE *s_dist    = (FLOAT_TYPE*) array;                // tpb distances
    int   *s_ID      = (int*)   &s_dist[blockDim.x];  // tpb IDs
    FLOAT_TYPE *s_ctr     = (FLOAT_TYPE*) &s_ID[blockDim.x];    // tpb centroid components
    
    unsigned int tid = threadIdx.x;                   // thread ID in block
    unsigned int t   = blockIdx.x * blockDim.x + tid; // global thread ID
    
    s_dist[tid] = 0.0;
    s_ID  [tid] = t;
    if (t < N)
    {
        // compute distance
        FLOAT_TYPE dist    = 0.0;
        int   offsetD = 0;
        // process each centroid component
        while (offsetD < D)
        {
            // read up to tpb components from global to shared memory
            if (offsetD + tid < D) s_ctr[tid] = CTR[offsetD + tid];
            __syncthreads();
            // compute distance in each dimension separately and build sum
            for (int d = 0; d < min(blockDim.x, D - offsetD); d++)
            {
                dist += distanceComponentGPU(s_ctr + d - offsetD, X + (offsetD + d) * N + t);
            }
            offsetD += blockDim.x;
            __syncthreads();
        }
        dist = distanceFinalizeGPU<FLOAT_TYPE>(1, &dist);
        // if new distance smaller then update assignment
        FLOAT_TYPE currDist = DIST[t];
        if (dist < currDist)
        {
            DIST  [t] = currDist = dist;
            ASSIGN[t] = iter;
        }
        s_dist[tid] = currDist;
    }
    __syncthreads();
    // find max distance of data point to centroid and its index in block
    parallelMax<THREADSPERBLOCK>(tid, s_dist, s_ID);
    // write maximum distance and index to global mem
    if (tid == 0) 
    {
        MAXDIST[blockIdx.x] = s_dist[tid];
        MAXID  [blockIdx.x] = s_ID[tid];
    }
}



void kcentersGPU(int N, int K, int D, FLOAT_TYPE *x, int *assign, FLOAT_TYPE *dist, int *centroids, int seed, DataIO *data)
{
    // CUDA kernel parameters
    int numBlock = (int) ceil((FLOAT_TYPE) N / (FLOAT_TYPE) THREADSPERBLOCK);
    dim3 block(THREADSPERBLOCK);
    dim3 grid(numBlock);
    int sMem = (2 * sizeof(FLOAT_TYPE) + sizeof(int)) * THREADSPERBLOCK;
    
    // GPU memory pointers, allocate and initialize device memory
    FLOAT_TYPE *dist_d         = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N, (void *)dist);
    int   *assign_d            = (int*)data->allocDeviceMemory(sizeof(int)   * N);
    FLOAT_TYPE *x_d            = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N * D, (void *)x);
    FLOAT_TYPE *ctr_d          = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * D);
    FLOAT_TYPE *maxDistBlock_d = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * numBlock);
    int   *maxIdBlock_d        = (int*)data->allocDeviceMemory(sizeof(int)   * numBlock);
    
    // Initialize host memory
    FLOAT_TYPE *maxDistBlock = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * numBlock);
    int   *maxID        = (int*)   malloc(sizeof(int));
    FLOAT_TYPE *ctr          = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * D);
    
    int centroid = seed;
    // for each cluster
    for (int k = 0; k < K; ++k)
    {
        // send centroid coordinates for current iteration to device memory
        for (int d = 0; d < D; d++) ctr[d] = x[d * N + centroid];
        cudaMemcpy(ctr_d, ctr, sizeof(FLOAT_TYPE) * D, cudaMemcpyHostToDevice);      
        centroids[k] = centroid;
        
        // for each data point, check if new centroid closer than previous best and if, reassign
        checkCentroid_CUDA<<<grid, block, sMem>>>(N, D, k, centroid, x_d, ctr_d, dist_d, assign_d, maxDistBlock_d, maxIdBlock_d);
        
        // get next max distance between data point and centroid
        cudaMemcpy(maxDistBlock, maxDistBlock_d, sizeof(FLOAT_TYPE) * numBlock, cudaMemcpyDeviceToHost);        
        int tempMax = 0;
        for (int i = 1; i < numBlock; i++) 
        {
            if (maxDistBlock[i] > maxDistBlock[tempMax]) tempMax = i;
        }
        cudaMemcpy(maxID, maxIdBlock_d + tempMax, sizeof(int), cudaMemcpyDeviceToHost);
        centroid = maxID[0];
    }
    // copy final assignments back to host
    cudaMemcpy(assign, assign_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
    
    // free up memory
    cudaFree(dist_d);
    cudaFree(assign_d);
    cudaFree(x_d);
    cudaFree(ctr_d);
    cudaFree(maxDistBlock_d);
    cudaFree(maxIdBlock_d);
    free(maxDistBlock);
    free(maxID);
    free(ctr);
}


