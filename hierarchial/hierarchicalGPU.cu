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
 * Authors: Kai J. Kohlhoff                                                    * 
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
 * \file hierarchicalGPU.cu
 * \brief A CUDA hierarchical clustering implementation
 *
 * Implements hierarchical clustering on the GPU
 *
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 **/

#ifdef HAVE_CONFIG_H
#include "../config.h"
#include "../campaign.h"
#else
#include "hierarchicalGPU.h"
#endif

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


__global__ static void calcDistanceMatrix_CUDA(int N, int D, FLOAT_TYPE *X, int *NEIGHBOR, FLOAT_TYPE *NEIGHBORDIST)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;  // global thread ID
    
    if (t < ceilf((FLOAT_TYPE) N / 2.0))
    {
        int row = t, col = -1, row2 = N - t - 1;
        NEIGHBOR[row] = NEIGHBOR[row2] = -1; 
        NEIGHBORDIST[row] = NEIGHBORDIST[row2] = FLT_MAX;
        // for each data point (of smaller index)
        // calculate lower diagonal matrix, each thread calculates N - 1 distances:
        // first, t for thread t, then (N - t - 1) for thread N - t - 1
        for (int j = 0; j < N - 1; j++)
        {
            col++;
            if (t == j) { row = row2; col = 0; }
            FLOAT_TYPE distance = 0.0;
            // compute distance
            for (int d = 0; d < D; d++) distance += distanceComponentGPU(X + d * N + row, X + d * N + col);
            distance = distanceFinalizeGPU(1, &distance);
            // update closest neighbor info if necessary
            if (distance < NEIGHBORDIST[row])
            {
                NEIGHBOR[row] = col;
                NEIGHBORDIST[row] = distance;
            }
        }
    }
}


__global__ void min_CUDA(unsigned int N, unsigned int iter, FLOAT_TYPE *INPUT, int *INKEY, FLOAT_TYPE *OUTPUT, int *OUTKEY)
{
    extern __shared__ FLOAT_TYPE array[];                        // declare variable in shared memory
    FLOAT_TYPE* s_value = (FLOAT_TYPE*) array;                        // allocate array of size tpb for values at offset 0
    int*   s_key   = (int*)   &s_value[blockDim.x];         // allocate array of size tpb for keys at offset tpb
    
    unsigned int tid = threadIdx.x;                         // in-block thread ID
    unsigned int t   = blockIdx.x*blockDim.x + threadIdx.x; // global thread ID
    s_value[tid] = FLT_MAX;
    s_key  [tid] = 0;
    
    // transfer data from global to shared memory
    if (t < N)
    {
        s_value[tid] = INPUT[t];
        s_key  [tid] = (iter == 0) ? t : INKEY[t];
    }
    __syncthreads();
    
    // do reduction in shared memory
    reduceMinTwo<THREADSPERBLOCK>(tid, s_value, s_key);
    
    // write result for block to global mem
    if (tid == 0)
    {
        OUTPUT[blockIdx.x] = s_value[tid];
        OUTKEY[blockIdx.x] = s_key  [tid];
    }
}



int getMinDistIndex(int numClust, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES)
{
    // prepare CUDA parameters
    dim3 block(THREADSPERBLOCK);
    int numBlocks = (int) ceil((FLOAT_TYPE) numClust / (FLOAT_TYPE) THREADSPERBLOCK);
    dim3 gridN(numBlocks);
    int sMem = (sizeof(FLOAT_TYPE) + sizeof(int)) * THREADSPERBLOCK;
    
    // cooperatively reduce distance array to find block-wise minimum and index
    min_CUDA<<<gridN, block, sMem>>>(numClust, 0, DISTS, INDICES, REDUCED_DISTS, REDUCED_INDICES);
    
    // repeat reduction until single value and key pair left
    while (numBlocks > 1)
    {
        int nElements = numBlocks;
        numBlocks = (int) ceil((FLOAT_TYPE) nElements / (FLOAT_TYPE) THREADSPERBLOCK);
        dim3 nBlocks(numBlocks);
        
        min_CUDA<<<nBlocks, block, sMem>>>(nElements, 1, REDUCED_DISTS, REDUCED_INDICES, REDUCED_DISTS, REDUCED_INDICES);
    }
    
    // copy result, i.e. index of element with minimal distance, to host
    int *ind = (int*) malloc(sizeof(int));
    cudaMemcpy(ind, REDUCED_INDICES, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    int index = ind[0];
    
    // free memory
    free(ind);
    
    return index;
}


__global__ static void mergeElementsInsertAtA_CUDA(int N, int D, int indexA, int indexB, FLOAT_TYPE *X)
{
    // compute global thread number 
    int d = blockDim.x * blockIdx.x + threadIdx.x;
    if (d < D)
    {
        X[d * N + indexA] = (X[d * N + indexA] + X[d * N + indexB]) / 2.0;
    }
}



__global__ static void computeAllDistancesToA_CUDA(int N, int numClust, int D, int indexA, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES) // includes distance from old B to A, which will later be discarded
{
    int numThreads = blockDim.x; // number of threads in block
    
    // define shared mem for first part of reduction step
    extern __shared__ FLOAT_TYPE array[];                  // declare variable in shared memory
    FLOAT_TYPE* s_dist  = (FLOAT_TYPE*) array;                  // dynamically allocate FLOAT_TYPE array at offset 0 to hold intermediate distances
    int*   s_index = (int*  ) &s_dist[numThreads];    // dynamically allocate int array at offset blockDim.x to hold intermediate indices
    FLOAT_TYPE* s_posA  = (FLOAT_TYPE*) &s_index[numThreads];   // dynamically allocate memory for position of element A
    
    int t = numThreads * blockIdx.x + threadIdx.x;    // global thread ID
    int tid = threadIdx.x;                            // in-block thread ID
    
    if (t < numClust) s_dist[tid] = 0.0;
    
    // for each dimension
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb components of element at j into shared memory (non-coalesced global memory access)
        if (offsetD + tid < D) s_posA[tid] = X[(offsetD + tid) * N + indexA];
        __syncthreads();
        // compute distances between up to tpb clusters and cluster at j
        for (unsigned int d = offsetD; d < min(offsetD + numThreads, D); d++)
        {
            if (t < numClust) s_dist[tid] += distanceComponentGPU(X + d * N + t, s_posA + d - offsetD);
        }
        offsetD += numThreads;
        __syncthreads();
    }
    if (t < numClust) s_dist[tid] = distanceFinalizeGPU(1, s_dist + tid);
    s_index[tid] = t;
    __syncthreads();
    
    // for clusters in sequence after A
    if (t > indexA && t < numClust)
    {
        FLOAT_TYPE dist = DISTS[t];
        if (s_dist[tid] == dist) INDICES[t] = min(indexA, INDICES[t]);
        else if (s_dist[tid] < dist)
        {
            DISTS  [t] = s_dist[tid];
            INDICES[t] = indexA;
        }
    }
    if (t >= indexA) s_dist[tid] = FLT_MAX;
    __syncthreads();
    
    // find minimum distance in array and index of corresponding cluster
    reduceMinTwo<THREADSPERBLOCK>(tid, s_dist, s_index);
    
    // write result for this block to global mem
    if (tid == 0)
    {
        REDUCED_DISTS  [blockIdx.x] = s_dist[tid];
        REDUCED_INDICES[blockIdx.x] = s_index[tid];
    }
}


__global__ static void updateElementA_CUDA(int indexA, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES)
{
    DISTS  [indexA] = REDUCED_DISTS  [0];
    INDICES[indexA] = REDUCED_INDICES[0];
}



void updateDistanceAndIndexForCluster(int indexA, int sMem, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES) // set smallest dist from A
{
    dim3 block(THREADSPERBLOCK);
    // after pre-reduction there will now be ceil((indexA - 1) / THREADSPERBLOCK) distances and indices in the arrays
    int numBlocks = (int) ceil((FLOAT_TYPE) (indexA - 1) / (FLOAT_TYPE) THREADSPERBLOCK);
    // finish reduction on GPU
    while (numBlocks > 1)
    {
        dim3 nBlocks((int) ceil((FLOAT_TYPE) numBlocks / (FLOAT_TYPE) THREADSPERBLOCK));
        
        min_CUDA<<<nBlocks, block, sMem>>>(numBlocks, 1, REDUCED_DISTS, REDUCED_INDICES, REDUCED_DISTS, REDUCED_INDICES);
        
        numBlocks = nBlocks.x;
    }
    // update min distance and index for element A
    dim3 numB(1);
    
    updateElementA_CUDA<<<numB, numB>>>(indexA, DISTS, INDICES, REDUCED_DISTS, REDUCED_INDICES);
}



__global__ static void moveCluster_CUDA(int N, int D, int indexB, int indexN, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES)
{
    // compute global thread number 
    int d = blockDim.x * blockIdx.x + threadIdx.x;
    // move the coordinates
    if (d < D)
    {
        X[d * N + indexB] = X[d * N + indexN];
    }
    // move the neighbor heuristic information
    if (d == 0)
    {
        DISTS  [indexB] = DISTS  [indexN];
        INDICES[indexB] = INDICES[indexN];
    }
}


__global__ static void computeDistancesToBForPLargerThanB_CUDA(int N, int D, int indexB, int numElements, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES)
{
    int numThreads = blockDim.x; // number of threads in block
    
    // define shared mem for first part of reduction step
    extern __shared__ FLOAT_TYPE array[];                 // declare variable in shared memory
    FLOAT_TYPE* s_posB = (FLOAT_TYPE*) array;                  // dynamically allocate memory for position of element B
    
    int t = numThreads * blockIdx.x + threadIdx.x;   // global thread ID
    int tid = threadIdx.x;                           // in-block thread ID
    
    FLOAT_TYPE dist = 0.0;
    
    // for each dimension
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb components of element at j into shared memory (non-coalesced global memory access)
        if (offsetD + tid < D) s_posB[tid] = X[(offsetD + tid) * N + indexB];
        __syncthreads();
        // compute distances between up to tpb clusters and cluster at j
        for (unsigned int d = offsetD; d < min(offsetD + numThreads, D); d++)
        {
            if (t < numElements) dist += distanceComponentGPU(X + d * N + (indexB + 1) + t, s_posB + d - offsetD);
        }
        offsetD += numThreads;
        __syncthreads();
    }
    if (t < numElements) dist = distanceFinalizeGPU(1, &dist);
    
    // t runs from 0 to (N - 1 - indexB)
    if (t < numElements)
    {
        int indexP = (t + indexB + 1);
        if (dist < DISTS[indexP])
        {
            DISTS  [indexP] = dist;
            INDICES[indexP] = indexB;
        }
    }
}



__global__ static void recomputeMinDistanceForElementAt_j_CUDA(int N, int D, int indexJ, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES)
{
    int numThreads = blockDim.x; // number of threads in block
    
    // allocate shared memory, requires 2*tpb*sizeof(FLOAT_TYPE) + tpb*sizeof(int)
    extern __shared__ FLOAT_TYPE array[];                // declare variable in shared memory
    FLOAT_TYPE* s_dist  = (FLOAT_TYPE*) array;                // dynamically allocate FLOAT_TYPE array at offset 0 to hold intermediate distances
    int*   s_index = (int*  ) &s_dist[numThreads];  // dynamically allocate int array at offset blockDim.x to hold intermediate indices
    FLOAT_TYPE* s_posJ  = (FLOAT_TYPE*) &s_index[numThreads]; // dynamically allocate memory for tpb components of element at j
    
    int t = numThreads * blockIdx.x + threadIdx.x;  // global thread ID
    int tid = threadIdx.x;                          // in-block thread ID
    
    s_dist[tid] = FLT_MAX;
    if (t < indexJ) s_dist[tid] = 0.0;
    
    // for each dimension
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb components of element at j into shared memory (non-coalesced global memory access)
        if (offsetD + tid < D) s_posJ[tid] = X[(offsetD + tid) * N + indexJ];
        __syncthreads();
        // compute distances between up to tpb clusters and cluster at j
        for (unsigned int d = offsetD; d < min(offsetD + numThreads, D); d++)
        {
            if (t < indexJ) s_dist[tid] += distanceComponentGPU(X + d * N + t, s_posJ + d - offsetD);
        }
        offsetD += numThreads;
        __syncthreads();
    }
    if (t < indexJ) 
    {
        s_dist[tid] = distanceFinalizeGPU(1, s_dist + tid);
        s_index[tid] = t;
    }
    __syncthreads();
    // find minimum distance in array and index of corresponding cluster
    reduceMinTwo<THREADSPERBLOCK>(tid, s_dist, s_index);
    
    // write result for this block to global mem
    if (tid == 0)
    {
        REDUCED_DISTS  [blockIdx.x] = s_dist[tid];
        REDUCED_INDICES[blockIdx.x] = s_index[tid];
    }  
}



int* hierarchicalGPU(int N, int D, FLOAT_TYPE *x, DataIO *data)
{
    // CUDA kernel parameters
    dim3 block(THREADSPERBLOCK);  
    int numBlocks  = (int) ceil((FLOAT_TYPE) N / (FLOAT_TYPE) THREADSPERBLOCK);
    int numBlocksD = (int) ceil((FLOAT_TYPE) D / (FLOAT_TYPE) THREADSPERBLOCK);
    dim3 gridN(numBlocks);
    dim3 gridD(numBlocksD);
    int sMemReduce = (sizeof(FLOAT_TYPE) + sizeof(int)) * THREADSPERBLOCK;
    
    // GPU memory pointers, allocate and initialize device memory
    FLOAT_TYPE *x_d             = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * D * N, (void *)x);
    int   *clustID_d       = (int*)data->allocDeviceMemory(sizeof(int) * N);           // indices of clusters
    int   *closestClust_d  = (int*)data->allocDeviceMemory(sizeof(int) * N);           // list of nearest neighbor indices
    FLOAT_TYPE *closestDist_d   = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N);         // list of nearest neighbor distances
    
    FLOAT_TYPE *REDUCED_DISTS   = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * (numBlocks + 1));
    int   *REDUCED_INDICES = (int*)data->allocDeviceMemory(sizeof(int) * (numBlocks + 1));
    
    // initialize host memory
    int* seq          = (int*) malloc(sizeof(int) * (N - 1) * 2); // sequential list of indices of merged pairs
    int* clustID      = (int*) malloc(sizeof(int) * N);           // indices of clusters
    int* closestClust = (int*) malloc(sizeof(int) * N);           // list of nearest neighbor indices
    if (seq == NULL || clustID == NULL || closestClust == NULL)
    {
        cout << "Error in hierarchicalCPU(): Unable to allocate sufficient memory" << endl;
        exit(1);
    }
    
    unsigned int posA, posB, last, nextID = N - 1;
    
    // implement neighbor heuristic
    // first step: compute all N^2 distances (i.e. N * (N - 1) / 2 distances using symmetry)
    //             save closest neighbor index and distance: O(N^2)
    calcDistanceMatrix_CUDA<<<gridN, block>>>(N, D, x_d, closestClust_d, closestDist_d);
    
    // copy closest cluster indices back to host
    cudaMemcpy(closestClust, closestClust_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
    
    // initialize clustID
    for (int i = 0; i < N; i++) clustID[i] = i;
    
    last = N;
    // pick and merge pair of clusters, repeat N-1 times
    for (int i = 0; i < N - 1; i++)
    {
        last--; // decrement counter to ignore last element
        nextID++;
        int newNumBlocks = (int) ceil((FLOAT_TYPE) last / (FLOAT_TYPE) THREADSPERBLOCK);
        dim3 newGridN(newNumBlocks);
        // require shared memory for distances (tpb FLOAT_TYPES), clustID (tpb ints), and element A (tpb FLOAT_TYPES) 
        int sMem = sizeof(FLOAT_TYPE) * 2 * THREADSPERBLOCK + sizeof(int) * THREADSPERBLOCK;
        
        cudaMemcpy(clustID_d, clustID, sizeof(int) * (last + 1), cudaMemcpyHostToDevice);
        
        // step1: get clustID for minimum distance
        // posB = getMinDistIndex(last + 1, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
        posB = getMinDistIndex(last + 1, closestDist_d, clustID_d, REDUCED_DISTS, REDUCED_INDICES);
        // get cluster ID of nearest neighbor
        posA = closestClust[posB];
        
        // update sequence of merged clusters
        seq[2 * i] = clustID[posA]; seq[2 * i + 1] = clustID[posB];
        
        // step2: merge elements and insert at A, update distances to A as necessary
        mergeElementsInsertAtA_CUDA<<<gridD, block>>>(N, D, posA, posB, x_d);
        
        clustID[posA] = nextID;
        
        if (posA != 0) // no distances for first array position
        {
            // compute distances from A to preceding clusters and from following clusters to A
            computeAllDistancesToA_CUDA<<<newGridN, block, sMem>>>(N, last, D, posA, x_d, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES); 
            
            // update nearest neighbor heuristic for A
            updateDistanceAndIndexForCluster(posA, sMemReduce, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
        }
        
        // step3: replace cluster at B by last cluster
        moveCluster_CUDA<<<gridD, block>>>(N, D, posB, last, x_d, closestDist_d, closestClust_d);
        
        // move cluster ID
        clustID[posB] = clustID[last];
        
        // check if index of element N still relevant
        if (closestClust[last] < posB) closestClust[posB] = closestClust[last];
        else closestClust[posB] = -1;  // this means that distance will be updated below
        if (last > posB)
        {
            dim3 gridLargerThanB((int) ceil((FLOAT_TYPE) (last - posB) / (FLOAT_TYPE) THREADSPERBLOCK)); // require (last - posB) threads
            int sMem2 = sizeof(FLOAT_TYPE) * THREADSPERBLOCK;
            
            // check if new cluster at B changes stored values for neighbor heuristic for following clusters
            computeDistancesToBForPLargerThanB_CUDA<<<gridLargerThanB, block, sMem2>>>(N, D, posB, last - posB, x_d, closestDist_d, closestClust_d);
        }
        
        // step4: look for elements at positions > A that have I or B as nearest neighbor and recalculate distances if found
        // for each element at position larger A check if it had A or B as neighbor and if, recompute
        // there is some redundancy possible, in case the new neighbor is the new A as this would already have been determined above
        for (int j = posA + 1; j < last; j++)
        {
            int neighbor = closestClust[j];
            // Attention: uses original neighbor assignment; on device, for all elements that previously had element A as clusest cluster, the neighbors have been set to -1
            if (neighbor == posA || neighbor == -1 || neighbor == posB)
            {
                int numBlocksJ = (int) ceil((FLOAT_TYPE) j / (FLOAT_TYPE) THREADSPERBLOCK);
                dim3 gridJ(numBlocksJ);
                
                // update neighbor heuristic for cluster at j by checking all preceding clusters
                recomputeMinDistanceForElementAt_j_CUDA<<<gridJ, block, sMem>>>(N, D, j, x_d, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
                
                // update nearest neighbor heuristic for j
                updateDistanceAndIndexForCluster(j, sMemReduce, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
            }
        }
        cudaMemcpy(closestClust, closestClust_d, sizeof(int) * last, cudaMemcpyDeviceToHost);
    }
    
    // free memory
    cudaFree(x_d);
    cudaFree(clustID_d);
    cudaFree(closestClust_d);
    cudaFree(closestDist_d);
    cudaFree(REDUCED_DISTS);
    cudaFree(REDUCED_INDICES);
    free(clustID);
    free(closestClust);
    
    return seq;
}

