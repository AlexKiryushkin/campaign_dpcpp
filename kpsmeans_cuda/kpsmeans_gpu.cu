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
 * \File kpsmeansGPU.cu
 * \brief A CUDA K-means implementation with all-prefix sorting and updating
 *
 * A module of the CAMPAIGN data clustering library for parallel architectures
 * Implements parallel K-means clustering with parallel 
 * all-prefix sum sorting and updating (Kps-means) on the GPU
 * 
 * \author Author: Kai Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 12/2/2010
 * \version 1.0
 **/

#include "kpsmeans_gpu.h"

#define FLOAT_TYPE float

using namespace std;

template <unsigned int BLOCKSIZE, class T>
__device__ static void reduceOne(int tid, T *s_A)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) { s_A[tid] += s_A[tid + 512]; } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) { s_A[tid] += s_A[tid + 256]; } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) { s_A[tid] += s_A[tid + 128]; } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) { s_A[tid] += s_A[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { s_A[tid] += s_A[tid + 32]; } __syncthreads();
        if (BLOCKSIZE >= 32) { s_A[tid] += s_A[tid + 16]; } __syncthreads();
        if (BLOCKSIZE >= 16) { s_A[tid] += s_A[tid +  8]; } __syncthreads();
        if (BLOCKSIZE >=  8) { s_A[tid] += s_A[tid +  4]; } __syncthreads();
        if (BLOCKSIZE >=  4) { s_A[tid] += s_A[tid +  2]; } __syncthreads();
        if (BLOCKSIZE >=  2) { s_A[tid] += s_A[tid +  1]; } __syncthreads();
    }
}


template <unsigned int BLOCKSIZE, class T, class U>
__device__ static void reduceTwo(int tid, T *s_A, U *s_B)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) { s_A[tid] += s_A[tid + 512]; s_B[tid] += s_B[tid + 512]; } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) { s_A[tid] += s_A[tid + 256]; s_B[tid] += s_B[tid + 256]; } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) { s_A[tid] += s_A[tid + 128]; s_B[tid] += s_B[tid + 128]; } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) { s_A[tid] += s_A[tid +  64]; s_B[tid] += s_B[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { s_A[tid] += s_A[tid + 32]; s_B[tid] += s_B[tid + 32]; } __syncthreads();
        if (BLOCKSIZE >= 32) { s_A[tid] += s_A[tid + 16]; s_B[tid] += s_B[tid + 16]; } __syncthreads();
        if (BLOCKSIZE >= 16) { s_A[tid] += s_A[tid +  8]; s_B[tid] += s_B[tid +  8]; } __syncthreads();
        if (BLOCKSIZE >=  8) { s_A[tid] += s_A[tid +  4]; s_B[tid] += s_B[tid +  4]; } __syncthreads();
        if (BLOCKSIZE >=  4) { s_A[tid] += s_A[tid +  2]; s_B[tid] += s_B[tid +  2]; } __syncthreads();
        if (BLOCKSIZE >=  2) { s_A[tid] += s_A[tid +  1]; s_B[tid] += s_B[tid +  1]; } __syncthreads();
    }
}



__global__ static void assignToClusters_KPSMCUDA(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN)
{
    extern __shared__ FLOAT_TYPE array[];                        // shared memory
    FLOAT_TYPE *s_center = (FLOAT_TYPE*) array;                       // tpb centroid components
    
    unsigned int t = blockDim.x * blockIdx.x + threadIdx.x; // global thread ID
    unsigned int tid = threadIdx.x;                         // thread ID in block
    
    // for each element
    if (t < N)
    {
        FLOAT_TYPE minDist  = 0.0;
        int   minIndex = 0;
        // for each centroid
        for (unsigned int k = 0; k < K; k++)
        {
            // compute distance
            FLOAT_TYPE dist = 0.0;
            unsigned int offsetD = 0;
            // loop over all dimensions in segments of size tpb
            while (offsetD < D)
            {
                // read up to tpb dimensions of centroid K (coalesced)
                if (offsetD + tid < D) s_center[tid] = CTR[k * D + offsetD + tid];
                __syncthreads();
                // for each of the following tpb (or D - offsetD) dimensions
                for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
                {
                    // broadcast centroid position and compute distance to data 
                    // point along dimension; reading of X is coalesced  
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + t));
                }
                offsetD += blockDim.x;
                __syncthreads();
            }
            dist = distanceFinalizeGPU<FLOAT_TYPE>(1, &dist);
            // if distance to centroid smaller than previous best, reassign
            if (dist < minDist || k == 0)
            {
                minDist = dist;
                minIndex = k;
            }
        }
        // now write index of closest centroid to global mem (coalesced)
        ASSIGN[t] = minIndex;
    }
}



__global__ static void calcScoreSorted_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, FLOAT_TYPE *SCORE, int *SEGOFFSET)
{
    extern __shared__ FLOAT_TYPE array[];                     // shared memory
    FLOAT_TYPE *s_scores  = (FLOAT_TYPE*) array;                   // tpb partial scores
    FLOAT_TYPE *s_center  = (FLOAT_TYPE*) &s_scores[blockDim.x];   // up to tpb centroid components
    int   *s_segment = (int*)   &s_center[blockDim.x];   // intermediate storage for broadcasting segment boundaries
    
    int k   = blockIdx.x;                                // cluster ID
    int tid = threadIdx.x;                               // in-block thread ID
    
    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];    // transfer start and end segment offsets to shared memory
    __syncthreads();
    
    int endOffset   = s_segment[1];                      // broadcast segment end offset to registers of all threads
    
    // initialize partial scores
    s_scores[tid] = 0.0;
    // loop over segment
    unsigned int offsetN = s_segment[0] + tid;
    while (offsetN < endOffset)
    {
        FLOAT_TYPE dist = 0.0;
        unsigned int offsetD = 0;
        // loop over dimensions
        while (offsetD < D)
        {
            // at each iteration read up to tpb centroid components from global mem (coalesced)
            if (offsetD + tid < D) s_center[tid] = CTR[k * D + offsetD + tid];
            __syncthreads();
            // for each of the following tpb (or D - offsetD) dimensions
            for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
            {
                // broadcast centroid component and compute contribution to distance to data point
                dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + offsetN));
            }
            offsetD += blockDim.x;
            __syncthreads();
        }
        // update partial score
        s_scores[tid] += distanceFinalizeGPU(1, &dist);
        offsetN += blockDim.x;
    }
    __syncthreads();
    // compute score for block by reducing over threads
    reduceOne<THREADSPERBLOCK>(tid, s_scores);
    if (tid == 0) SCORE[k] = s_scores[tid];
}


__global__ static void calcScore_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN, FLOAT_TYPE *SCORE)
{
    extern __shared__ FLOAT_TYPE array[];                     // shared memory
    FLOAT_TYPE *s_scores = (FLOAT_TYPE*) array;                    // tpb partial scores
    FLOAT_TYPE *s_center = (FLOAT_TYPE*) &s_scores[blockDim.x];    // up to tpb centroid components
    
    int k   = blockIdx.x;                                // cluster ID
    int tid = threadIdx.x;                               // in-block thread ID
    
    // initialize partial scores
    s_scores[tid] = 0.0;
    // loop over data points
    unsigned int offsetN = tid;
    while (offsetN < N)
    {
        FLOAT_TYPE dist = 0.0;
        unsigned int offsetD = 0;
        // loop over dimensions
        while (offsetD < D)
        {
            // at each iteration read up to tpb centroid components from global mem (coalesced)
            if (offsetD + tid < D) s_center[tid] = CTR[k * D + offsetD + tid];
            __syncthreads();
            // thread divergence likely
            if (ASSIGN[offsetN] == k)
            {
                // for each of the following tpb (or D - offsetD) dimensions
                for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
                {
                    // broadcast centroid component and compute contribution to distance to data point
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + offsetN));
                }
            }
            offsetD += blockDim.x;
            __syncthreads();
        }
        // update partial score
        s_scores[tid] += distanceFinalizeGPU(1, &dist);
        offsetN += blockDim.x;
    }
    __syncthreads();
    // compute score for block by reducing over threads
    reduceOne<THREADSPERBLOCK>(tid, s_scores);
    if (tid == 0) SCORE[k] = s_scores[tid];
}



__global__ static void calcCentroidsSorted_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *SEGOFFSET)
{
    extern __shared__ FLOAT_TYPE array[];                            // shared memory
    int   *s_numElements = (int*)   array;                      // tpb partial sums of elements in cluster
    FLOAT_TYPE *s_centerParts = (FLOAT_TYPE*) &s_numElements[blockDim.x]; // tpb partial centroid components
    int   *s_segment     = (int*)   &s_centerParts[blockDim.x];
    
    int k   = blockIdx.x;                                       // centroid ID
    int tid = threadIdx.x;                                      // in-block thread ID
    
    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];           // transfer start and end offsets to shared memory
    __syncthreads();
    int writeOffset = s_segment[0] + tid;                       // broadcast segment start offset to registers of all threads and add unique tid
    int endOffset   = s_segment[1];                             // broadcast segment end offset to registers of all threads
    
    FLOAT_TYPE clusterSize = 0.0;                                    // only used by thread 0 in block
    
    // initialize partial cluster size
    s_numElements[tid] = 0;
    // for each dimension
    for (unsigned int d = 0; d < D; d++)
    {
        // initialize centroid parts
        s_centerParts[tid] = 0.0;
        unsigned int offset = writeOffset;
        // loop over segment
        while (offset < endOffset)
        {
            // update centroid parts
            s_centerParts[tid] += X[d * N + offset];
            // increment number of elements
            if (d == 0) s_numElements[tid]++;
            // move on to next segment
            offset += blockDim.x;
        }
        __syncthreads();
        
        // take sum over all tpb array elements
        // reduce number of cluster elements only once
        if (d == 0)
        {
            // reduce number of elements and centroid parts
            reduceTwo<THREADSPERBLOCK>(tid, s_centerParts, s_numElements);
            if (tid == 0) clusterSize = (FLOAT_TYPE) s_numElements[tid];
        }
        else 
        {
            // reduce centroid parts
            reduceOne<THREADSPERBLOCK>(tid, s_centerParts);
        }
        // write result to global mem (non-coalesced)
        // could replace this by coalesced writes followed by matrix transposition
        if (tid == 0) if (clusterSize > 0) CTR[k * D + d] = s_centerParts[tid] / clusterSize;
    }
}



__global__ static void calcCentroids_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN)
{
    extern __shared__ FLOAT_TYPE array[];                            // shared memory
    int   *s_numElements = (int*)   array;                      // tpb partial sums of elements in cluster
    FLOAT_TYPE *s_centerParts = (FLOAT_TYPE*) &s_numElements[blockDim.x]; // tpb partial centroid components
    
    int k   = blockIdx.x;                                       // cluster ID
    int tid = threadIdx.x;                                      // in-block thread ID
    
    FLOAT_TYPE clusterSize = 0.0;                                    // only used by thread 0 in block
    
    // initialize partial cluster size
    s_numElements[tid] = 0;
    // for each dimension
    for (unsigned int d = 0; d < D; d++)
    {
        // initialize centroid parts
        s_centerParts[tid] = 0.0;
        unsigned int offset = tid;
        // loop over data points
        while (offset < N)
        {
            // thread divergence likely
            if (ASSIGN[offset] == k)
            {
                // update centroid parts
                s_centerParts[tid] += X[d * N + offset];
                // increment number of elements in cluster
                if (d == 0) s_numElements[tid]++;
            } 
            // move on to next segment
            offset += blockDim.x;
        }
        __syncthreads();
        
        // take sum over all tpb array elements
        // reduce number of cluster elements only once
        if (d == 0)
        {
            // reduce number of elements and centroid parts
            reduceTwo<THREADSPERBLOCK>(tid, s_centerParts, s_numElements);
            if (tid == 0) clusterSize = (FLOAT_TYPE) s_numElements[tid];
            // note: if clusterSize == 0 we can return here
        }
        else 
        {
            // reduce centroid parts
            reduceOne<THREADSPERBLOCK>(tid, s_centerParts);
        }
        // write result to global mem (non-coalesced)
        // could replace this by coalesced writes followed by matrix transposition
        if (tid == 0) if (clusterSize > 0) CTR[k * D + d] = s_centerParts[tid] / clusterSize;
    }
}



template <unsigned int BLOCKSIZE>
__device__ static int parallelPrefixSum(int tid, int *DATA)
{
    unsigned int temp = 0;
    unsigned int sum  = 0;
    unsigned int n    = 2 * BLOCKSIZE;    // always work with 2 * tpb;
    
    // parallel reduction
    if (n >= 1024) { if (tid < 512) { DATA[tid] += DATA[tid + 512]; } __syncthreads(); }
    if (n >=  512) { if (tid < 256) { DATA[tid] += DATA[tid + 256]; } __syncthreads(); }
    if (n >=  256) { if (tid < 128) { DATA[tid] += DATA[tid + 128]; } __syncthreads(); }
    if (n >=  128) { if (tid <  64) { DATA[tid] += DATA[tid +  64]; } __syncthreads(); }
    
    if (tid <  32) DATA[tid] += DATA[tid + 32];
    __syncthreads();
    if (tid <  16) DATA[tid] += DATA[tid + 16];
    __syncthreads();
    if (tid <   8) DATA[tid] += DATA[tid +  8];
  __syncthreads();
    if (tid <   4) DATA[tid] += DATA[tid +  4];
    __syncthreads();
    if (tid <   2) DATA[tid] += DATA[tid +  2];
    __syncthreads();
    if (tid <   1) DATA[tid] += DATA[tid +  1];
    
    __syncthreads();
    // broadcast and store reduced sum of elements in array across threads
    sum = DATA[0];
    __syncthreads();
    if (tid == 0) DATA[0] = 0;
    
    // parallel all-prefix sum using intermediate results from reduction
    if (tid <   1) { temp = DATA[tid]; DATA[tid] += DATA[tid +   1]; DATA[tid +   1] = temp; }
    __syncthreads();
    if (tid <   2) { temp = DATA[tid]; DATA[tid] += DATA[tid +   2]; DATA[tid +   2] = temp; }
    __syncthreads();
    if (tid <   4) { temp = DATA[tid]; DATA[tid] += DATA[tid +   4]; DATA[tid +   4] = temp; }
    __syncthreads();
    if (tid <   8) { temp = DATA[tid]; DATA[tid] += DATA[tid +   8]; DATA[tid +   8] = temp; }
    __syncthreads();
    if (tid <  16) { temp = DATA[tid]; DATA[tid] += DATA[tid +  16]; DATA[tid +  16] = temp; }
    __syncthreads();
    if (tid <  32) { temp = DATA[tid]; DATA[tid] += DATA[tid +  32]; DATA[tid +  32] = temp; }
    __syncthreads();
    if (n >=  128) { if (tid <  64) { temp = DATA[tid]; DATA[tid] += DATA[tid +  64]; DATA[tid +  64] = temp; } __syncthreads(); }
    if (n >=  256) { if (tid < 128) { temp = DATA[tid]; DATA[tid] += DATA[tid + 128]; DATA[tid + 128] = temp; } __syncthreads(); }
    if (n >=  512) { if (tid < 256) { temp = DATA[tid]; DATA[tid] += DATA[tid + 256]; DATA[tid + 256] = temp; } __syncthreads(); }
    if (n >= 1024) { if (tid < 512) { temp = DATA[tid]; DATA[tid] += DATA[tid + 512]; DATA[tid + 512] = temp; } __syncthreads(); }
    
    return sum;
}



__global__ static void sort_getSegmentSize_CUDA(int N, int *ASSIGN, int *SEGSIZE)
{
    // requires tpb*sizeof(int) bytes of shared memory  
    extern __shared__ FLOAT_TYPE array[];                   // shared memory  
    int   *s_num    = (int*)   array;                  // tpb partial segment sizes
    
    unsigned int k   = blockIdx.x;                     // cluster ID
    unsigned int tid = threadIdx.x;                    // thread ID in block
    unsigned int num = 0;
    
    // loop over data points
    unsigned int offsetN = tid;
    while (offsetN < N)
    {
        // add up data points assigned to cluster (coalesced global memory reads)
        if (ASSIGN[offsetN] == k) num++;
        offsetN += THREADSPERBLOCK;
    }
    // collect partial sums in shared memory
    s_num[tid] = num;
    __syncthreads();
    // now reduce sum over threads in block
    reduceOne<THREADSPERBLOCK>(tid, s_num);
    // output segment size for cluster
    if (tid == 0) SEGSIZE[k] = s_num[tid];
}


__global__ static void sort_moveData_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *X2, int *ASSIGN, int *UPDASSIGN, int *SEGOFFSET)
{
    // requires (5*tpb+2)*sizeof(int) Bytes of shared memory
    extern __shared__ FLOAT_TYPE array[];                            // shared memory
    int  *s_gather  = (int*)   array;                           // up to 3 * tpb assignments
    int  *s_indices = (int*)   &s_gather[3 * THREADSPERBLOCK];  // 2 * tpb intermediate results for parallel all-prefix sum
    int *s_segment  = (int*)   &s_indices[2 * THREADSPERBLOCK]; // 2 integer numbers to hold start and end offsets of segment for broadcasting
    
    bool scan1 = false;
    bool scan2 = false;
    
    int k   = blockIdx.x;                                       // cluster ID
    int tid = threadIdx.x;                                      // in-block thread ID
    s_gather[tid                      ] = 0;
    s_gather[tid +     THREADSPERBLOCK] = 0;
    s_gather[tid + 2 * THREADSPERBLOCK] = 0;
    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];           // transfer start and end offsets to shared memory
    __syncthreads();
    int bufferOffset = s_segment[0] + tid;                      // broadcast segment start offset to registers of all threads and add unique tid
    int bufferEndOffset   = s_segment[1];                       // broadcast segment end offset to registers of all threads
    int dataOffset = tid;                                       // offset in data set
    int windowSize = 2 * THREADSPERBLOCK;                       // number of data points processed per iterations
    int numFound   = 0;                                         // number of data points collected
    
    // Notes:
    // uses parallel string compaction to collect data points assigned to cluster
    // works with segments of size windowSize = 2*tpb to make use of all threads for prefix sum computation
    // collects indices for data points assigned to cluster and moves data to buffer once windowSize indices are available
    // checks assignments for all N data points in ceil(N/windowSize) iterations
    // completes when all data points for k have been found
    while ((dataOffset - tid) < N && (bufferOffset - tid + numFound) < bufferEndOffset)
    {
        // convert windowSize assignments to binary: 0 = not assigned, or 1 = assigned to current cluster
        
        // data point at tid assigned to cluster?
        scan1 = ((dataOffset                       < N) && (ASSIGN[dataOffset]                       == k));
        // not yet end of data array and data point at tid + tpb assigned to cluster?
        scan2 = ((dataOffset +     THREADSPERBLOCK < N) && (ASSIGN[dataOffset +     THREADSPERBLOCK] == k));
        // set shared memory array to 1 for data points in segment assigned to cluster k, and to 0 otherwise
        s_indices[tid                  ] = 0;
        s_indices[tid + THREADSPERBLOCK] = 0;
        if (scan1) s_indices[tid                  ] = 1;
        if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;
        
        // returns unique indices for data points assigned to cluster and total number of data points found in segment
        int nonZero = parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices);
        
        // check which (if any) data points belong to cluster and store indices of those found
        if (scan1) s_gather[numFound + s_indices[tid                  ]] = dataOffset;
        if (scan2) s_gather[numFound + s_indices[tid + THREADSPERBLOCK]] = dataOffset + THREADSPERBLOCK;
        // update number of data points found but not yet transferred to buffer
        numFound += nonZero;
        
        __syncthreads();
        // while we have enough indices to do efficient coalesced memory writes
        while (numFound >= THREADSPERBLOCK)
        {
            // for each dimension
            for (unsigned int d = 0; d < D; d++)
            {
                // copy data point from data array (non-coalesced read) to buffer (coalesced write)
                X2[bufferOffset + d * N] = X[s_gather[tid] + d * N];
            }
            // update assignment
            UPDASSIGN[bufferOffset] = k;
            // update number of data points found but not yet transferred
            numFound -= THREADSPERBLOCK;
            // move down collected indices of data points and overwrite those already used 
            s_gather[tid                      ] = s_gather[tid +     THREADSPERBLOCK];
            s_gather[tid +     THREADSPERBLOCK] = s_gather[tid + 2 * THREADSPERBLOCK];
            s_gather[tid + 2 * THREADSPERBLOCK] = 0;
            // move to next buffer segment
            bufferOffset += THREADSPERBLOCK;
        }
        // move to next data segment
        dataOffset += windowSize;
        __syncthreads();
    }
    // now write the remaining data points to buffer (writing coaelesced, but not necessarily all threads involved)
    if (bufferOffset < bufferEndOffset)
    {
        // for each dimension, transfer data
        for (unsigned int d = 0; d < D; d++) X2[bufferOffset + d * N] = X[s_gather[tid] + d * N];
        // update assignment
        UPDASSIGN[bufferOffset] = k; 
    }
}


// Attention: CPU version
void serialPrefixSum_KPSMCUDA(int N, int *INPUT, int *OUTPUT)
{
    // transfer data to host
    int *intermediates = (int*)  malloc(sizeof(int) * N);
    cudaMemcpy(intermediates, INPUT, sizeof(int) * N, cudaMemcpyDeviceToHost); 
    // value at location 0 is okay
    for (unsigned int i = 1; i < N; i++)
    { 
        intermediates[i] += intermediates[i - 1];
    }
    // transfer results to device
    cudaMemcpy(OUTPUT, intermediates, sizeof(int) * N, cudaMemcpyHostToDevice);
    free(intermediates);
}



__global__ static void upd_getReassigns_CUDA(int N, int *ASSIGN, int *SEGOFFSET, int *SEGSIZE, int *UPDSEGSIZE)
{
    // requires (tpb+2)*sizeof(int) bytes of shared memory 
    extern __shared__ FLOAT_TYPE array[];                        // shared memory
    int   *s_num     = (int*)   array;                      // tpb partial sums of wrongly assigned data points
    int   *s_segment = (int*)   &s_num[THREADSPERBLOCK];    // intermediate storage for broadcasting segment boundaries
    
    unsigned int k   = blockIdx.x;                          // cluster ID
    unsigned int tid = threadIdx.x;                         // in-block thread ID
    unsigned int num = 0;
    
    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];       // transfer start and end offsets to shared memory
    __syncthreads();
    int endOffset   = s_segment[1];                         // broadcast segment end offset to registers of all threads
    
    // loop over data segment
    unsigned int offsetN = s_segment[0] + tid;              // broadcast segment start offset and add unique tid
    while (offsetN < endOffset)
    {
        // add up wrongly assigned (coalesced global memory reads)
        if (ASSIGN[offsetN] != k) num++;
        offsetN += THREADSPERBLOCK;
    }
    // collect partial sum in shared memory
    s_num[tid] = num;
    __syncthreads();
    // now reduce sum over threads in block
    reduceOne<THREADSPERBLOCK>(tid, s_num);
    if (tid == 0) 
    {
        // output segment size for block
        UPDSEGSIZE[k] = s_num[tid];
        // update data segment size for block
        SEGSIZE[k] -= s_num[tid];
    }
}



__global__ static void upd_moveAssigns_CUDA(int N, int *ASSIGN, int *UPDASSIGN, int *SEGOFFSET, int *UPDSEGOFFSET)
{
    // requires (5*tpb+4)*sizeof(int) Bytes of shared memory
    extern __shared__ FLOAT_TYPE array[];                            // shared memory
    int  *s_gather  = (int*)   array;                           // up to 3 * tpb assignments
    int  *s_indices = (int*)   &s_gather[3 * THREADSPERBLOCK];  // 2 * tpb intermediate results for parallel all-prefix sum
    int *s_segment  = (int*)   &s_indices[2 * THREADSPERBLOCK]; // 4 integer numbers to hold start and end offsets of segments for broadcasting
    
    bool scan1 = false;
    bool scan2 = false;
    
    int k   = blockIdx.x;                                       // cluster ID
    int tid = threadIdx.x;                                      // in-block thread ID
    
    s_gather[tid                      ] = 0;
    s_gather[tid +     THREADSPERBLOCK] = 0;
    s_gather[tid + 2 * THREADSPERBLOCK] = 0;
    if (tid < 2) 
    {
        s_segment[tid] = UPDSEGOFFSET[k + tid];                 // transfer start and end offset of buffer segment to shared memory
        s_segment[tid + 2] = SEGOFFSET[k + tid];                // transfer start and end offset of assignment segment to shared memory
    }
    __syncthreads();
    int bufferOffset = s_segment[0] + tid;                      // broadcast segment start offset to registers of all threads and add unique tid
    int bufferEndOffset  = s_segment[1];                        // broadcast segment end offset to registers of all threads
    int dataOffset = s_segment[2] + tid;                        // broadcast assignment start offset
    int dataEndOffset = s_segment[3];                           // broadcast assignment end offset
    int windowSize = 2 * THREADSPERBLOCK;                       // number of data points processed per iteration
    int numFound   = 0;                                         // number of data points collected
    
    // completes when all misassigned data points are found
    while ((dataOffset - tid) < dataEndOffset && (bufferOffset - tid + numFound) < bufferEndOffset)
    {
        // collect tpb indices using string compaction, then move data
        // convert windowSize assignments to binary: 1 = not assigned, or 0 = assigned to current cluster
        
        // data point at tid not assigned to cluster?
        scan1 = ((dataOffset                   < dataEndOffset) && (ASSIGN[dataOffset] != k));
        // not yet end of array and data point at tid + tpb not assigned to cluster?
        scan2 = ((dataOffset + THREADSPERBLOCK < dataEndOffset) && (ASSIGN[dataOffset + THREADSPERBLOCK] != k));
        // set shared memory to 1 for data points not assigned to cluster k, and to 0 otherwise
        s_indices[tid                  ] = 0;
        s_indices[tid + THREADSPERBLOCK] = 0;
        if (scan1) s_indices[tid                  ] = 1;
        if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;
        
        // returns unique indices for data points not assigned to cluster and total number of data points found in segment
        int nonZero = parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices);
        
        // check which (if any) data points do not belong to cluster and store indices of those found
        if (scan1) s_gather[numFound + s_indices[tid]] = dataOffset;
        if (scan2) s_gather[numFound + s_indices[tid + THREADSPERBLOCK]] = dataOffset + THREADSPERBLOCK;
        // update number of data points found but not yet transferred to buffer
        numFound += nonZero;
        
        __syncthreads();
        // while we have enough indices to do efficient coalesced memory writes
        while (numFound >= THREADSPERBLOCK)
        {
            // transfer assignment to buffer
            UPDASSIGN[bufferOffset] = ASSIGN[s_gather[tid]];
            // update number of assignments found but not yet transferred
            numFound -= THREADSPERBLOCK;
            // move down collected indices of assignments and overwrite those already used
            s_gather[tid                      ] = s_gather[tid +     THREADSPERBLOCK];
            s_gather[tid +     THREADSPERBLOCK] = s_gather[tid + 2 * THREADSPERBLOCK];
            s_gather[tid + 2 * THREADSPERBLOCK] = 0;
            // move to next buffer segment
            bufferOffset += THREADSPERBLOCK;
        }      
        // move to next data segment
        dataOffset += windowSize;
        __syncthreads();
    }
    // now write the remaining assignments to buffer (writing coalesced, but not necessarily complete set of threads involved)
    if (bufferOffset < bufferEndOffset)
    {
        // transfer assignment
        UPDASSIGN[bufferOffset] = ASSIGN[s_gather[tid]];
    }
}



__global__ static void upd_segmentSize_CUDA(int *UPDASSIGN, int *BUFFERSIZE, int *SEGSIZE)
{
    // requires (tpb+1)*sizeof(int) bytes of shared memory
    extern __shared__ FLOAT_TYPE array[];                       // shared memory  
    int   *s_num     = (int*)   array;                     // tpb partial sums of assignments to cluster k in buffer
    int   *s_segment = (int*)   &s_num[THREADSPERBLOCK];      
    
    unsigned int k   = blockIdx.x;                         // cluster ID
    unsigned int tid = threadIdx.x;                        // thread ID in block
    unsigned int num = 0;
    
    if (tid == 0) s_segment[tid] = BUFFERSIZE[0];          // transfer end offset to shared memory
    __syncthreads();
    int endOffset   = s_segment[0];                        // broadcast segment end offset to registers of all threads
    
    // loop over data points
    unsigned int offsetN = tid;
    while (offsetN < endOffset)
    {
        // add up assignments to cluster (coalesced global memory reads)
        if (UPDASSIGN[offsetN] == k) num++;
        // move on to next buffer segment
        offsetN += THREADSPERBLOCK;
    }
    // collect partial sums in shared memory
    s_num[tid] = num;
    __syncthreads();
    // now reduce sum over threads in block
    reduceOne<THREADSPERBLOCK>(tid, s_num);
    // output new data segment size for cluster
    if (tid == 0) SEGSIZE[k] += s_num[tid];
}



__global__ static void upd_findWrongAssigns_CUDA(int N, int *ASSIGN, int *SEGOFFSET, int *UPDSEGSIZE)
{
    // requires (tpb+2)*sizeof(int) bytes of shared memory
    extern __shared__ FLOAT_TYPE array[];                         // shared memory  
    int   *s_num     = (int*)   array;                       // tpb partial segment sizes
    int   *s_segment = (int*)   &s_num[THREADSPERBLOCK];     // intermediate storage for broadcasting segment boundaries
    
    unsigned int k   = blockIdx.x;                           // cluster ID
    unsigned int tid = threadIdx.x;                          // in-block thread ID
    unsigned int num = 0;
    
    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];        // transfer start and end offsets to shared memory
    __syncthreads();
    int endOffset   = s_segment[1];                          // broadcast segment end offset to registers of all threads
    
    // loop over segment
    unsigned int offsetN = s_segment[0] + tid;
    while (offsetN < endOffset)
    {
        // add up incorrectly assigned data points (coalesced global memory reads)
        if (ASSIGN[offsetN] != k) num++;
        offsetN += THREADSPERBLOCK;
    }
    // collect partial sums in shared memory
    s_num[tid] = num;
    __syncthreads();
    // now reduce sum over threads in block                                       
    reduceOne<THREADSPERBLOCK>(tid, s_num);
    // output segment size for cluster
    if (tid == 0) UPDSEGSIZE[k] = s_num[tid];
}



__global__ static void upd_moveDataToBuffer_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *X2, int *ASSIGN, int *UPDASSIGN, int *SEGOFFSET, int *UPDSEGOFFSET)
{
    // requires (5*tpb+4)*sizeof(int) Bytes of shared memory
    extern __shared__ FLOAT_TYPE array[];                            // shared memory
    int  *s_gather  = (int*)   array;                           // up to 3 * tpb assignments
    int  *s_indices = (int*)   &s_gather[3 * THREADSPERBLOCK];  // 2 * tpb intermediate results for parallel all-prefix sum
    int *s_segment  = (int*)   &s_indices[2 * THREADSPERBLOCK]; // 4 integer numbers to hold start and end offsets of buffer and data segments for broadcasting
    
    bool scan1 = false;
    bool scan2 = false;
    
    int k   = blockIdx.x;                                       // cluster ID
    int tid = threadIdx.x;                                      // in-block thread ID
    s_gather[tid                      ] = 0;
    s_gather[tid +     THREADSPERBLOCK] = 0;
    s_gather[tid + 2 * THREADSPERBLOCK] = 0;
    if (tid < 2) 
    {
        s_segment[tid] = UPDSEGOFFSET[k + tid];                 // transfer start and end offsets of buffer segment to shared memory
        s_segment[tid + 2] = SEGOFFSET[k + tid];                // transfer start and end offsets of data segment to shared memory
    }
    __syncthreads();
    int bufferOffset = s_segment[0] + tid;                      // broadcast buffer segment start offset to registers of all threads and add unique tid
    int bufferEndOffset   = s_segment[1];                       // broadcast buffer segment end offset to registers of all threads
    int dataOffset = s_segment[2] + tid;                        // offset in data set
    int dataEndOffset = s_segment[3];
    int windowSize = 2 * THREADSPERBLOCK;                       // number of data points processed per iterations
    int numFound   = 0;                                         // number of data points collected
    
    // completes when all misassigned data points have been found
    while ((dataOffset - tid) < dataEndOffset && (bufferOffset - tid + numFound) < bufferEndOffset)
    {
        // collect tpb indices using string compaction, then move data
        // move windowSize assignments to binary: 1 = not assigned, or 0 = assigned to current cluster
        
        // data point at tid not assigned to cluster?
        scan1 = ((dataOffset                       < dataEndOffset) && (ASSIGN[dataOffset]                       != k));
        // not yet end of array and data point at tid + tpb not assigned to cluster?
        scan2 = ((dataOffset +     THREADSPERBLOCK < dataEndOffset) && (ASSIGN[dataOffset +     THREADSPERBLOCK] != k));
        // set shared memory to 1 for data points not assigned to cluster k, and to 0 otherwise
        s_indices[tid                  ] = 0;
        s_indices[tid + THREADSPERBLOCK] = 0;
        if (scan1) s_indices[tid                  ] = 1;
        if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;
        
        // returns unique indices for data points not assigned to cluster and total number of data points found in segment
        int nonZero = parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices);
        
        // check which (if any) data points do not belong to cluster and store indices of those found
        if (scan1) s_gather[numFound + s_indices[tid                  ]] = dataOffset;
        if (scan2) s_gather[numFound + s_indices[tid + THREADSPERBLOCK]] = dataOffset + THREADSPERBLOCK;
        // update number of data points found but not yet transferred to buffer
        numFound += nonZero;
        
        __syncthreads();
        // while we have enough indices to do efficient coalesced memory writes
        while (numFound >= THREADSPERBLOCK)
        {
            // transfer data points to buffer
            for (unsigned int d = 0; d < D; d++) X2[bufferOffset + d * N] = X[s_gather[tid] + d * N];
            // transfer assignment to buffer
            UPDASSIGN[bufferOffset] = ASSIGN[s_gather[tid]]; 
            // update number of data points found but not yet transferred
            numFound -= THREADSPERBLOCK;
            // move down collected indices of assignments and overwrite those already used
            s_gather[tid                      ] = s_gather[tid +     THREADSPERBLOCK];
            s_gather[tid +     THREADSPERBLOCK] = s_gather[tid + 2 * THREADSPERBLOCK];
            s_gather[tid + 2 * THREADSPERBLOCK] = 0;
            // move to next buffer segment
            bufferOffset += THREADSPERBLOCK;
        }
        // move to next data segment
        dataOffset += windowSize;
        __syncthreads();
    }
    // now write the remaining data points to buffer (writing coaelesced, but not necessarily all threads involved)
    if (bufferOffset < bufferEndOffset)
    {
        // transfer data point
        for (int d = 0; d < D; d++) X2[bufferOffset + d * N] = X[s_gather[tid] + d * N];
        // transfer assignment
        UPDASSIGN[bufferOffset] = ASSIGN[s_gather[tid]];
    }
}



__global__ static void upd_collectDataFromBuffer_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *X2, int *ASSIGN, int *UPDASSIGN, int *SEGOFFSET, int *BUFFERSIZE)
{
    // requires (8*tpb)*sizeof(int) Bytes of shared memory
    extern __shared__ FLOAT_TYPE array[];                              // shared memory
    int  *s_gatherA   = (int*)   array;                           // up to 3 * tpb assignments
    int  *s_gatherB   = (int*)   &s_gatherA[3 * THREADSPERBLOCK]; // up to 3 * tpb assignments
    int  *s_indices   = (int*)   &s_gatherB[3 * THREADSPERBLOCK]; // 2 * tpb intermediate results for parallel all-prefix sum
    
    bool scan1 = false;
    bool scan2 = false;
    
    int k   = blockIdx.x;                                         // cluster ID
    int tid = threadIdx.x;                                        // in-block thread ID
    s_gatherA[tid                      ] = 0;
    s_gatherA[tid +     THREADSPERBLOCK] = 0;
    s_gatherA[tid + 2 * THREADSPERBLOCK] = 0;
    s_gatherB[tid                      ] = 0;
    s_gatherB[tid +     THREADSPERBLOCK] = 0;
    s_gatherB[tid + 2 * THREADSPERBLOCK] = 0;
    if (tid < 2) s_indices[tid] = SEGOFFSET[k + tid];             // transfer start and end offsets of data segment to shared memory
    if (tid < 1) s_indices[tid + 2] = BUFFERSIZE[0];              // transfer end offset of buffer segment to shared memory
    __syncthreads();
    int bufferOffset = tid;                                       // start at first position of buffer
    int bufferEndOffset = s_indices[2];                           // broadcast buffer end offset to registers of all threads
    int dataOffset = s_indices[0] + tid;                          // broadcast data start offset to registers of all threads and add unique tid
    int dataEndOffset = s_indices[1];                             // broadcast data end offset to registers of all threads
    __syncthreads();
    int numFoundData= 0;                                          // number of gaps in data array found
    int numFoundBuffer = 0;                                       // number of data points in buffer found
    
    // run through segment assigned to current cluster center and fill in the missing data points
    while ((dataOffset - tid) < dataEndOffset || (bufferOffset - tid) < bufferEndOffset)
    {
        // collect tpb indices from data array and tpb indices from buffer using string compaction, then move data
        // move windowSize assignments to shared memory coded as 0 (not assigned to current cluster) or 1 (assigned)
        while ((dataOffset - tid) < dataEndOffset && numFoundData < THREADSPERBLOCK)
        {
            // data point at tid not assigned to cluster?
            scan1 = ((dataOffset                   < dataEndOffset) && (ASSIGN[dataOffset]                   != k));
            // not yet end of array and data point at tid + tpb not assigned to cluster?
            scan2 = ((dataOffset + THREADSPERBLOCK < dataEndOffset) && (ASSIGN[dataOffset + THREADSPERBLOCK] != k));
            // set shared memory to 1 for data points not assigned to cluster k, and to 0 otherwise
            s_indices[tid                  ] = 0;
            s_indices[tid + THREADSPERBLOCK] = 0;
            if (scan1) s_indices[tid                  ] = 1;
            if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;
            
            // returns unique indices for data points not assigned to cluster and total number of such data points found in segment
            int nonZero = parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices);
            
            // check which (if any) data points do not belong to cluster and store indices of those found
            if (scan1) s_gatherA[numFoundData + s_indices[tid                  ]] = dataOffset;
            if (scan2) s_gatherA[numFoundData + s_indices[tid + THREADSPERBLOCK]] = dataOffset + THREADSPERBLOCK;
            // update number of data points found but not yet transferred from buffer
            numFoundData += nonZero;
            // move to next data segment
            dataOffset += 2 * THREADSPERBLOCK;
        }
        // one of two options: we have collected enough indices to do memory transfer using all threads, or we have reached the end of the segment
        __syncthreads();
        
        // now, do the same for the buffer data
        while ((bufferOffset - tid) < bufferEndOffset && numFoundBuffer < THREADSPERBLOCK)
        {
            // data point at tid assigned to cluster?
            scan1 = ((bufferOffset                   < bufferEndOffset) && (UPDASSIGN[bufferOffset]                   == k));
            // not yet end of array and data point at tid + tpb assigned to cluster?
            scan2 = ((bufferOffset + THREADSPERBLOCK < bufferEndOffset) && (UPDASSIGN[bufferOffset + THREADSPERBLOCK] == k));
            // set shared memory to 1 for data points assigned to cluster k, and to 0 otherwise
            s_indices[tid                  ] = 0;
            s_indices[tid + THREADSPERBLOCK] = 0;
            if (scan1) s_indices[tid                  ] = 1;
            if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;
            
            // returns unique indices for data points assigned to cluster and total number of such data points found in segment
            int nonZero = parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices);
            
            // check which (if any) data points belong to cluster and store indices of those found
            if (scan1) s_gatherB[numFoundBuffer + s_indices[tid                  ]] = bufferOffset;
            if (scan2) s_gatherB[numFoundBuffer + s_indices[tid + THREADSPERBLOCK]] = bufferOffset + THREADSPERBLOCK;
            // update number of data points found but not yet transferred to data array
            numFoundBuffer += nonZero;
            // move to next buffer segment
            bufferOffset += 2 * THREADSPERBLOCK;
        }
        // one of two options: we have collected enough indices to do memory transfer using all threads, or we have reached the end of the buffer
        
        __syncthreads();
        
        // do we have enough indices to do efficient coalesced memory writes
        if (numFoundData >= THREADSPERBLOCK && numFoundBuffer >= THREADSPERBLOCK)
        {
            // overwrite wrongly assigned data point in data segment with data point from buffer
            for (unsigned int d = 0; d < D; d++) X[s_gatherA[tid] + d * N] = X2[s_gatherB[tid] + d * N];
            // update assignment for data point just overwritten
            ASSIGN[s_gatherA[tid]] = k;
            // update number of data points found but not yet transferred from buffer to data array
            numFoundData -= THREADSPERBLOCK;
            numFoundBuffer -= THREADSPERBLOCK;
            // move down collected indices and overwrite those already used
            s_gatherA[tid                      ] = s_gatherA[tid +     THREADSPERBLOCK];
            s_gatherB[tid                      ] = s_gatherB[tid +     THREADSPERBLOCK];
            s_gatherA[tid +     THREADSPERBLOCK] = s_gatherA[tid + 2 * THREADSPERBLOCK];
            s_gatherB[tid +     THREADSPERBLOCK] = s_gatherB[tid + 2 * THREADSPERBLOCK];
            s_gatherA[tid + 2 * THREADSPERBLOCK] = 0;
            s_gatherB[tid + 2 * THREADSPERBLOCK] = 0;
        }
        __syncthreads();
    }
    // Note: numFoundData != numFoundBuffer would be an error
    // now write the remaining data points to buffer (neither reading nor writing coaelesced, not necessarily all threads involved)
    if (tid < numFoundData)
    {
        // transfer data point
        for (int d = 0; d < D; d++) X[s_gatherA[tid] + d * N] = X2[s_gatherB[tid] + d * N];
        // update assignment
        ASSIGN[s_gatherA[tid]] = k;
    }
}


bool updateData(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *X2, int *ASSIGN, int *UPDASSIGN, int *SEGSIZE, int *UPDSEGSIZE, int *SEGOFFSET, int *UPDSEGOFFSET)
{
    dim3 block(THREADSPERBLOCK); // set number of threads per block
    dim3 gridK(K);               // K blocks of threads in grid
    
    // step 1: Find number of data points wrongly assigned and update number of data points, updates SEGSIZE and UPDSEGSIZE
    int sMem = sizeof(int) * (THREADSPERBLOCK + 2);
    upd_getReassigns_CUDA<<<gridK, block, sMem>>>(N, ASSIGN, SEGOFFSET, SEGSIZE, UPDSEGSIZE); 
    
    // step 2: Determine offsets to move assignments of wrongly assigned data points to buffer 
    serialPrefixSum_KPSMCUDA(K, UPDSEGSIZE, UPDSEGOFFSET + 1);
    
    // step 3: Use string compaction to collect indices of wrongly assigned data points (moves assignments only)
    sMem = sizeof(int)*(5*THREADSPERBLOCK + 4);
    upd_moveAssigns_CUDA<<<gridK, block, sMem>>>(N, ASSIGN, UPDASSIGN, SEGOFFSET, UPDSEGOFFSET);
    
    // step 4: Updates number of data points per segment by checking wrongly assigned for new ones
    sMem = sizeof(int) * (THREADSPERBLOCK + 1);
    upd_segmentSize_CUDA<<<gridK, block, sMem>>>(UPDASSIGN, UPDSEGOFFSET + K, SEGSIZE);
    
    // step 5: Determine new global offsets
    serialPrefixSum_KPSMCUDA(K, SEGSIZE, SEGOFFSET + 1);
    
    // step 6: With new global offsets determine how many elements are in wrong segment
    sMem = sizeof(int) * (THREADSPERBLOCK + 2);
    upd_findWrongAssigns_CUDA<<<gridK, block, sMem>>>(N, ASSIGN, SEGOFFSET, UPDSEGSIZE);
    
    // step 7: Check how many data points are wrongly assigned in total
    //         if that number >= N/2 then sort all
    //         if that number < 50% of N then update sort
    serialPrefixSum_KPSMCUDA(K, UPDSEGSIZE, UPDSEGOFFSET + 1);
    
    int *numReassigns = (int*) malloc(sizeof(int) * 1);
    cudaMemcpy(numReassigns, UPDSEGOFFSET + K, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    
    if (numReassigns[0] * 2 >= N)
    {
        // step 8a: Updating less efficient than full re-sort, return
        return false;
        // done
    }
    else
    {
        // step 8b: Write data to buffer
        sMem = sizeof(int)*(5*THREADSPERBLOCK + 4); 
        upd_moveDataToBuffer_CUDA<<<gridK, block, sMem>>>(N, D, X, X2, ASSIGN, UPDASSIGN, SEGOFFSET, UPDSEGOFFSET);
        
        // step 9: Check data in buffer and write to data array to fill gaps in segments
        sMem = sizeof(int)*(8*THREADSPERBLOCK);
        upd_collectDataFromBuffer_CUDA<<<gridK, block, sMem>>>(N, D, X, X2, ASSIGN, UPDASSIGN, SEGOFFSET, UPDSEGOFFSET + K); 
        return true;
        // done
    }
}


void sortData(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *X2, int *ASSIGN, int *UPDASSIGN, int *SEGSIZE, int *SEGOFFSET)
{
    dim3 block(THREADSPERBLOCK); // set number of threads per block
    dim3 gridK(K);               // K blocks of threads in grid
    
    // loop over all data points, detect those that are assigned to cluster k,
    // determine unique memory indices in contiguous stretch of shared memory 
    // using string compaction with parallel prefix sum, then move data to buffer
    
    // run over all data points and detect those assigned to cluster K
    int sMem = (sizeof(int) * THREADSPERBLOCK);
    sort_getSegmentSize_CUDA<<<gridK, block, sMem>>>(N, ASSIGN, SEGSIZE);
    
    // first segment offset is 0 (as of initialization), compute the others as running sum over segment sizes
    serialPrefixSum_KPSMCUDA(K, SEGSIZE, SEGOFFSET + 1);
    
    // now move the data from X to X2
    sMem = sizeof(int)*(5*THREADSPERBLOCK + 2);
    sort_moveData_CUDA<<<gridK, block, sMem>>>(N, D, X, X2, ASSIGN, UPDASSIGN, SEGOFFSET);
}



FLOAT_TYPE kpsmeansGPU(int N, int K, int D, FLOAT_TYPE *x, FLOAT_TYPE *ctr, int *assign, unsigned int maxIter, DataIO *data)
{
    // CUDA kernel parameters
    dim3 block(THREADSPERBLOCK);
    dim3 gridK(K); 
    dim3 gridN((int) ceil((FLOAT_TYPE) N / (FLOAT_TYPE) THREADSPERBLOCK));
    int sMemAssign  = (sizeof(FLOAT_TYPE) *     THREADSPERBLOCK);
    int sMemScore   = (sizeof(FLOAT_TYPE) * 2 * THREADSPERBLOCK);
    int sMemCenters = (sizeof(FLOAT_TYPE) *     THREADSPERBLOCK + sizeof(int) * THREADSPERBLOCK);
    
    // GPU memory pointers, allocate and initialize device memory
    FLOAT_TYPE *x_d         = (FLOAT_TYPE *)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N * D, (void *)x);
    FLOAT_TYPE *x2_d        = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(float) * N * D);
    FLOAT_TYPE *ctr_d       = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * K * D, (void *)ctr);
    int *segsize_d     = (int *)data->allocDeviceMemory(sizeof(int) * K);
    int *segoffs_d     = (int*)data->allocZeroedDeviceMemory(sizeof(int) * (K+1));
    int *upd_segsize_d = (int*)data->allocDeviceMemory(sizeof(int) * K);
    int *upd_segoffs_d = (int*)data->allocZeroedDeviceMemory(sizeof(int) * (K+1));
    int *assign_d      = (int*)data->allocDeviceMemory(sizeof(int) * N);
    int *upd_assign_d  = (int*)data->allocDeviceMemory(sizeof(int) * N);
    FLOAT_TYPE *s_d         = (FLOAT_TYPE*)data->allocZeroedDeviceMemory(sizeof(FLOAT_TYPE) * K);
    
    // Initialize host memory
    FLOAT_TYPE *s   = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * K);
    
    // initialize scores
    FLOAT_TYPE oldscore = -1000.0, score = 0.0;
    if (maxIter < 1) maxIter = INT_MAX;
    unsigned int iter = 0;
    bool sorted = false;
    // loop until defined number of iterations reached or converged
    while (iter < maxIter && ((score - oldscore) * (score - oldscore)) > EPS)
    {
        oldscore = score;
        
        // skip at first iteration and use provided centroids instead
        if (iter > 0) 
        {
            // for sorted data
            if (sorted) 
            {
                calcCentroidsSorted_CUDA<<<gridK, block, sMemCenters + 2 * sizeof(int)>>>(N, D, x_d, ctr_d, segoffs_d);
            }
            // for unsorted data
            else 
            {
                calcCentroids_CUDA<<<gridK, block, sMemCenters>>>(N, D, x_d, ctr_d, assign_d);
            }
            sorted = false;
        }
        iter++;
        
        // update clusters and create backup of current cluster centers
        assignToClusters_KPSMCUDA<<<gridN, block, sMemAssign>>>(N, K, D, x_d, ctr_d, assign_d);
        
        // if not first iteration try updating the partially sorted array
        if (DOSORTING && DOUPDATING && iter > 1)
        {
            sorted = updateData(N, K, D, x_d, x2_d, assign_d, upd_assign_d, segsize_d, upd_segsize_d, segoffs_d, upd_segoffs_d);
        }
        // if first iteration, or updating was not successful, perform full sorting of data
        if (DOSORTING && !sorted)
        {
            // sort the data and assignments to buffer
            sortData(N, K, D, x_d, x2_d, assign_d, upd_assign_d, segsize_d, segoffs_d);
            
            // swap sorted buffer with unsorted data
            FLOAT_TYPE *temp_d = x_d;
            x_d = x2_d;
            x2_d = temp_d;
            
            // swap old assignments with new assignments
            int *a_temp_d = assign_d;
            assign_d = upd_assign_d;
            upd_assign_d = a_temp_d;
            
            sorted = true;
        }
        
        // for sorted data
        if (sorted) 
        {
            // get score per cluster for sorted data
            calcScoreSorted_CUDA<<<gridK, block, sMemScore + 2 * sizeof(int)>>>(N, D, x_d, ctr_d, s_d, segoffs_d);
        }
        // for unsorted data
        else 
        {
            // get score per cluster for unsorted data
            calcScore_CUDA<<<gridK, block, sMemScore>>>(N, D, x_d, ctr_d, assign_d, s_d);
        }
        
        // copy scores per cluster back to the host and do reduction on CPU
        cudaMemcpy(s, s_d,    sizeof(FLOAT_TYPE) * K, cudaMemcpyDeviceToHost);  
        score = 0.0;
        for (int i = 0; i < K; i++) score += s[i];
    }
    cout << "Number of iterations: " << iter << endl;
    
    // copy centroids back to host
    cudaMemcpy(ctr, ctr_d,         sizeof(FLOAT_TYPE) * K * D, cudaMemcpyDeviceToHost);
    // copy assignments back to host
    cudaMemcpy(assign, assign_d,    sizeof(int)   * N    , cudaMemcpyDeviceToHost);
    
    // free memory
    cudaFree(x_d);
    cudaFree(x2_d);
    cudaFree(ctr_d);
    cudaFree(segsize_d);
    cudaFree(segoffs_d);
    cudaFree(upd_segsize_d);
    cudaFree(upd_segoffs_d);
    cudaFree(assign_d);
    cudaFree(upd_assign_d);
    cudaFree(s_d);
    free(s);
    
    return score;
}
