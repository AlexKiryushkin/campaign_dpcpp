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
 * \File kmedoidsGPU.cu
 * \brief A CUDA K-medoids implementation with all-prefix sorting
 *
 * A module of the CAMPAIGN data clustering library for parallel architectures
 * Implements parallel K-medoids clustering with parallel 
 * all-prefix sum sorting on the GPU
 * 
 * \author Author: Kai Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 15/2/2010
 * \version 1.0
 **/

#include "./kmedoidsGPU.h"

using namespace std;

/** 
 * \brief Parallel algorithm, reduction (sum of elements) of an array
 * Runtime O(log(BLOCKSIZE)) = O(1)
 * Works for up to 1024 elements in array
 * Called from within a kernel, will be inlined
 *
 * \param tid Thread ID
 * \param s_A Array in shared memory
 * \return Result of reduction in first element of array s_A
 */
template <unsigned int BLOCKSIZE, class T>
__device__ static void reduceOne(int tid, T *s_A)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512) { s_A[tid] += s_A[tid + 512]; } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256) { s_A[tid] += s_A[tid + 256]; } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128) { s_A[tid] += s_A[tid + 128]; } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64) { s_A[tid] += s_A[tid +  64]; } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { s_A[tid] += s_A[tid + 32]; }
        if (BLOCKSIZE >= 32) { s_A[tid] += s_A[tid + 16]; }
        if (BLOCKSIZE >= 16) { s_A[tid] += s_A[tid +  8]; }
        if (BLOCKSIZE >=  8) { s_A[tid] += s_A[tid +  4]; }
        if (BLOCKSIZE >=  4) { s_A[tid] += s_A[tid +  2]; }
        if (BLOCKSIZE >=  2) { s_A[tid] += s_A[tid +  1]; }
    }
}



__global__ static void assignToClusters_KMDCUDA(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN)
{
    extern __shared__ FLOAT_TYPE array[];                        // shared memory
    FLOAT_TYPE *s_center = (FLOAT_TYPE*) array;                       // tpb medoid components
    
    unsigned int t = blockDim.x * blockIdx.x + threadIdx.x; // global thread ID
    unsigned int tid = threadIdx.x;                         // thread ID in block
    
    // for each element
    if (t < N)
    {
        FLOAT_TYPE minDist  = 0.0;
        int   minIndex = 0;
        // for each medoid
        for (unsigned int k = 0; k < K; k++)
        {
            // compute distance
            FLOAT_TYPE dist = 0.0;
            unsigned int offsetD = 0;
            // loop over all dimensions in segments of size tpb
            while (offsetD < D)
            {
                // read up to tpb dimensions of medoid K (coalesced)
                if (offsetD + tid < D) s_center[tid] = CTR[k * D + offsetD + tid];
                __syncthreads();
                // for each of the following tpb (or D - offsetD) dimensions
                for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
                {
                    // broadcast medoid position and compute distance to data 
                    // point along dimension; reading of X is coalesced  
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + t));
                }
                offsetD += blockDim.x;
                __syncthreads();
            }
            dist = distanceFinalizeGPU<FLOAT_TYPE>(1, &dist);
            // if distance to medoid smaller than previous best, reassign
            if (dist < minDist || k == 0)
            {
                minDist = dist;
                minIndex = k;
            }
        }
        // now write index of closest medoid to global mem (coalesced)
        ASSIGN[t] = minIndex;
    }
}


__global__ static void calcScoreSorted_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, FLOAT_TYPE *SCORE, int *SEGOFFSET)
{
    extern __shared__ FLOAT_TYPE array[];                     // shared memory
    FLOAT_TYPE *s_scores  = (FLOAT_TYPE*) array;                   // tpb partial scores
    FLOAT_TYPE *s_center  = (FLOAT_TYPE*) &s_scores[blockDim.x];   // up to tpb medoid components
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
            // at each iteration read up to tpb medoid components from global mem (coalesced)
            if (offsetD + tid < D) s_center[tid] = CTR[k * D + offsetD + tid];
            __syncthreads();
            // for each of the following tpb (or D - offsetD) dimensions
            for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
            {
                // broadcast medoid component and compute contribution to distance to data point
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


__global__ static void calcNewScoreSortedAndSwap_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, FLOAT_TYPE *CTR2, int *INDEX, FLOAT_TYPE *SCORE, int *SEGOFFSET, int *MEDOID, int *RANDOM)
{
    extern __shared__ FLOAT_TYPE array[];                       // shared memory
    FLOAT_TYPE *s_scores  = (FLOAT_TYPE*) array;                     // tpb partial scores plus one for current medoid's score
    FLOAT_TYPE *s_center  = (FLOAT_TYPE*) &s_scores[blockDim.x + 1]; // up to tpb medoid components
    int   *s_segment = (int*)   &s_center[blockDim.x];     // intermediate storage for broadcasting segment boundaries
    int   *s_random  = (int*)   &s_segment[2];             // random number for cluster
    
    int k   = blockIdx.x;                                  // cluster ID
    int tid = threadIdx.x;                                 // in-block thread ID
    
    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];      // transfer start and end segment offsets to shared memory
    if (tid == 0) s_random[tid] = RANDOM[k];               // transfer random number to shared memory
    if (tid == 0) s_scores[blockDim.x + tid] = SCORE[k];   // transfer score for current medoid to shared memory
    __syncthreads();
    
    FLOAT_TYPE score = s_scores[blockDim.x];                    // broadcast score for current medoid
    int endOffset   = s_segment[1];                        // broadcast segment end offset to registers of all threads
    
    // initialize partial scores
    s_scores[tid] = 0.0;
    // start of segment
    unsigned int offsetN = s_segment[0];
    
    // compute index of new medoid using random number for cluster
    int newMedoid = offsetN + s_random[0] % (endOffset - offsetN);
    
    offsetN += tid;
    while (offsetN < endOffset)
    {
        FLOAT_TYPE dist = 0.0;
        unsigned int offsetD = 0;
        // loop over dimensions
        while (offsetD < D)
        {
            // at each iteration read up to tpb medoid components from global mem (non-coalesced!!!)
            if (offsetD + tid < D) 
            {
                s_center[tid] = X[(offsetD + tid) * N + newMedoid]; 
                // store medoid components in global mem (coalesced)
                CTR2[k * D + offsetD + tid] = s_center[tid];
            }
            __syncthreads();
            // for each of the following tpb (or D - offsetD) dimensions
            for (unsigned int d = offsetD; d < min(offsetD + blockDim.x, D); d++)
            {
                // broadcast medoid component and compute contribution to distance to data point
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
    
    // check if new score for cluster better than previous one; if it is, reassign medoid
    // also accept equal score to improve sampling
    if (s_scores[0] <= score)
    {
        // update score and medoid index
        if (tid == 0)
        {
            SCORE[k] = s_scores[tid];
            MEDOID[k] = INDEX[newMedoid];
        }
        // update medoid coordinates
        unsigned int offsetD = tid;
        while (offsetD < D)
        {
            // copy from buffer to coordinate array, reading and writing coalesced
            CTR[k * D + offsetD] = CTR2[k * D + offsetD];
            offsetD += blockDim.x;
        }
    }
}


// ************ SECTION FOR PARALLEL DATA SORTING *******************  



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
    if (tid <  16) DATA[tid] += DATA[tid + 16];
    if (tid <   8) DATA[tid] += DATA[tid +  8];
    if (tid <   4) DATA[tid] += DATA[tid +  4];
    if (tid <   2) DATA[tid] += DATA[tid +  2];
    if (tid <   1) DATA[tid] += DATA[tid +  1];
    
    __syncthreads();
    // broadcast and store reduced sum of elements in array across threads
    sum = DATA[0];
    __syncthreads();
    if (tid == 0) DATA[0] = 0;
    
    // parallel all-prefix sum using intermediate results from reduction
    if (tid <   1) { temp = DATA[tid]; DATA[tid] += DATA[tid +   1]; DATA[tid +   1] = temp; }
    if (tid <   2) { temp = DATA[tid]; DATA[tid] += DATA[tid +   2]; DATA[tid +   2] = temp; }
    if (tid <   4) { temp = DATA[tid]; DATA[tid] += DATA[tid +   4]; DATA[tid +   4] = temp; }
    if (tid <   8) { temp = DATA[tid]; DATA[tid] += DATA[tid +   8]; DATA[tid +   8] = temp; }
    if (tid <  16) { temp = DATA[tid]; DATA[tid] += DATA[tid +  16]; DATA[tid +  16] = temp; }
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



__global__ static void sort_moveData_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *X2, int *INDEX, int *INDEX2, int *ASSIGN, int *SEGOFFSET)
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
            // copy data point indices
            INDEX2[bufferOffset] = INDEX[s_gather[tid]];
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
        // copy data point indices
        INDEX2[bufferOffset] = INDEX[s_gather[tid]];
    }
}

/**
 * \brief Determines how many data points are assigned to each cluster
 *
 * \param N Size of array
 * \param INPUT Input array of size N in GPU global memory
 * \param OUTPUT Output array of size N + 1 in GPU global memory
 * \return Running sum over INPUT in OUTPUT
 */
// Attention: CPU version
void serialPrefixSum_KMDCUDA(int N, int *INPUT, int *OUTPUT)
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



void sortData(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *X2, int *INDEX, int *INDEX2, int *ASSIGN, int *SEGSIZE, int *SEGOFFSET)
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
    serialPrefixSum_KMDCUDA(K, SEGSIZE, SEGOFFSET + 1);
    
    // now move the data from X to X2
    sMem = sizeof(int)*(5*THREADSPERBLOCK + 2);
    sort_moveData_CUDA<<<gridK, block, sMem>>>(N, D, X, X2, INDEX, INDEX2, ASSIGN, SEGOFFSET);
}



FLOAT_TYPE kmedoidsGPU(int N, int K, int D, FLOAT_TYPE *x, int *medoid, int *assign, unsigned int maxIter, DataIO *data)
{
    // CUDA kernel parameters
    dim3 block(THREADSPERBLOCK);
    dim3 gridK(K); 
    dim3 gridN((int) ceil((FLOAT_TYPE) N / (FLOAT_TYPE) THREADSPERBLOCK));
    int sMemAssign    = (sizeof(FLOAT_TYPE) *     THREADSPERBLOCK);
    int sMemScore     = (sizeof(FLOAT_TYPE) * 2 * THREADSPERBLOCK + sizeof(int) * 2);
    int sMemScoreSwap = sMemScore + sizeof(FLOAT_TYPE) + sizeof(int);
    
    // Initialize host memory
    FLOAT_TYPE *ctr     = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * K * D); // Coordinates for K medoids
    int   *indices = (int*)   malloc(sizeof(int) * N);       // data point indices, shuffled by sorting
    int   *random  = (int*)   malloc(sizeof(int) * K);       // Array of random numbers
    FLOAT_TYPE *s       = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * K);     // K scores for K clusters
    
    // copy coordinates for initial set of medoids
    for (unsigned int k = 0; k < K; k++)
    {
        if (medoid[k] < 0 || medoid[k] >= N) 
        {
            cerr << "Error: medoid " << k << " (" << medoid[k] << ") does not map to data point" << endl;
            return 0.0;
        }
        for (unsigned int d = 0; d < D; d++) ctr[k * D + d] = x[d * N + medoid[k]];
    }
    
    // initialize data point indices
    for (unsigned int n = 0; n < N; n++) indices[n] = n;
    
    // GPU memory pointers, allocate and initialize device memory
    FLOAT_TYPE *x_d      = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N * D, (void *)x);
    FLOAT_TYPE *x2_d     = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N * D);
    FLOAT_TYPE *ctr_d    = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * K * D, (void *)ctr);
    FLOAT_TYPE *ctr2_d   = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * K * D);
    int *indices_d  = (int *)data->allocDeviceMemory(sizeof(int) * N, (void *)indices);
    int *indices2_d = (int*)data->allocDeviceMemory(sizeof(int) * N);
    int *random_d   = (int*)data->allocDeviceMemory(sizeof(int) * K);
    int *segsize_d  = (int*)data->allocDeviceMemory(sizeof(int) * K);
    int *segoffs_d  = (int*)data->allocZeroedDeviceMemory(sizeof(int) * (K+1));
    int *assign_d   = (int*)data->allocDeviceMemory(sizeof(int) * N);
    int *medoid_d   = (int*)data->allocDeviceMemory(sizeof(int) * K, (void *)medoid);
    FLOAT_TYPE *s_d      = (FLOAT_TYPE*)data->allocZeroedDeviceMemory(sizeof(FLOAT_TYPE) * K);
    
    // loop for defined number of iterations
    unsigned int iter = 0;
    while (iter < maxIter)
    {
        // assign data points to clusters
        assignToClusters_KMDCUDA<<<gridN, block, sMemAssign>>>(N, K, D, x_d, ctr_d, assign_d);
        
        // sort by assignment (O(K*D*N))
        sortData(N, K, D, x_d, x2_d, indices_d, indices2_d, assign_d, segsize_d, segoffs_d);
        
        // swap sorted buffer with unsorted data
        FLOAT_TYPE *dataTemp_d = x_d; x_d = x2_d; x2_d = dataTemp_d;
        
        // swap shuffled indices with those from previous iteration
        int *indTemp_d = indices_d; indices_d = indices2_d; indices2_d = indTemp_d;
        
        // get score per cluster for sorted data
        calcScoreSorted_CUDA<<<gridK, block, sMemScore>>>(N, D, x_d, ctr_d, s_d, segoffs_d);
        
        // generate K random numbers, do this on CPU
        for (unsigned int k = 0; k < K; k++) random[k] = rand();
        cudaMemcpy(random_d, random, sizeof(int) * K, cudaMemcpyHostToDevice);
        
        // compute score for randomly selected new set of medoids and swap medoids score improves
        calcNewScoreSortedAndSwap_CUDA<<<gridK, block, sMemScoreSwap>>>(N, D, x_d, ctr_d, ctr2_d, indices_d, s_d, segoffs_d, medoid_d, random_d);
        
        iter++;
    } 
    
    // copy scores per cluster back to the host and do reduction on CPU
    cudaMemcpy(s, s_d,    sizeof(FLOAT_TYPE) * K, cudaMemcpyDeviceToHost);  
    FLOAT_TYPE score = 0.0;
    for (int i = 0; i < K; i++) score += s[i];
    
    // copy medoids back to host
    cudaMemcpy(medoid, medoid_d, sizeof(int) * K      , cudaMemcpyDeviceToHost);
    // copy medoid coordinates back to host
    cudaMemcpy(ctr, ctr_d,       sizeof(FLOAT_TYPE) * K * D, cudaMemcpyDeviceToHost);
    // copy assignments back to host
    cudaMemcpy(assign, assign_d, sizeof(int)   * N    , cudaMemcpyDeviceToHost);
    
    // free memory
    cudaFree(x_d);
    cudaFree(x2_d);
    cudaFree(ctr_d);
    cudaFree(ctr2_d);
    cudaFree(random_d);
    cudaFree(segsize_d);
    cudaFree(segoffs_d);
    cudaFree(assign_d);
    cudaFree(s_d);
    cudaFree(medoid_d);
    free(ctr);
    free(random);
    free(s);
    
    return score;
}

