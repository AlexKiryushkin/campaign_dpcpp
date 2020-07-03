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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "./kmedoidsGPU.h"
#include <cmath>

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
static void reduceOne(int tid, T *s_A,
                      sycl::nd_item<3> item_ct1)
{
  for (unsigned idx{ 1024U }; idx > 1; idx /= 2)
  {
    const auto halfIdx = idx / 2u;
    if (BLOCKSIZE >= idx) 
    {
      if (tid < halfIdx)
      {
        s_A[tid] += s_A[tid + halfIdx];
      }
      item_ct1.barrier();
    }
  }
}



static void assignToClusters_KMDCUDA(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN,
                                     sycl::nd_item<3> item_ct1,
                                     uint8_t *dpct_local)
{
  auto * array    = (float*)dpct_local;
  auto * s_center = (FLOAT_TYPE*)array;

  unsigned int t = item_ct1.get_global_id(2);
  unsigned int tid = item_ct1.get_local_id(2);

  const auto processElement = t < N;
  FLOAT_TYPE minDist = 0.0;
  int   minIndex = 0;
  for (unsigned int k = 0; k < K; k++)
  {
    FLOAT_TYPE dist = 0.0;
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
      if (offsetD + tid < D)
      {
        s_center[tid] = CTR[k * D + offsetD + tid];
      }
      item_ct1.barrier();
      
      const auto minValue = sycl::min((unsigned int)(offsetD + item_ct1.get_local_range(2)), (unsigned int)D);
      for (unsigned int d = offsetD; d < minValue; d++)
      {
        if (processElement)
        {
          dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + t));
        }
      }
      offsetD += item_ct1.get_local_range(2);
      item_ct1.barrier();
    }
    dist = distanceFinalizeGPU<FLOAT_TYPE>(1, &dist);
    if (processElement && ((dist < minDist) || (k == 0)))
    {
      minDist = dist;
      minIndex = k;
    }
  }
  
  if (processElement)
  {
    ASSIGN[t] = minIndex;
  }
}


static void calcScoreSorted_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, FLOAT_TYPE *SCORE, int *SEGOFFSET,
                                 sycl::nd_item<3> item_ct1,
                                 uint8_t *dpct_local)
{
  auto * array     = (float*)dpct_local;
  auto * s_scores  = (FLOAT_TYPE*)array;
  auto * s_center  = (FLOAT_TYPE*)&s_scores[item_ct1.get_local_range(2)];
  auto * s_segment = (int*)&s_center[item_ct1.get_local_range(2)];

  int k = item_ct1.get_group(2);
  int tid = item_ct1.get_local_id(2);

  if (tid < 2)
  {
    s_segment[tid] = SEGOFFSET[k + tid];
  }
  item_ct1.barrier();

  int endOffset = s_segment[1];

  s_scores[tid] = 0.0;
  int startOffset = s_segment[0];
  unsigned int offsetN = startOffset + tid;

  const auto iterationCount = 
    (sycl::abs(endOffset - startOffset) + item_ct1.get_local_range(2) - 1) / item_ct1.get_local_range(2);
  for (auto idx{ 0U }; idx < iterationCount; ++idx)
  {
    const auto processElement = offsetN < endOffset;
    FLOAT_TYPE dist = 0.0;
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
      if (offsetD + tid < D)
      {
        s_center[tid] = CTR[k * D + offsetD + tid];
      }
      item_ct1.barrier();
      
      const auto minValue = sycl::min( (unsigned int)(offsetD + item_ct1.get_local_range(2)), (unsigned int)D);
      for (unsigned int d = offsetD; d < minValue; d++)
      {
        if (processElement)
        {
          dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + offsetN));
        }
      }
      offsetD += item_ct1.get_local_range(2);
      item_ct1.barrier();
    }
    // update partial score
    s_scores[tid] += distanceFinalizeGPU(1, &dist);
    offsetN += item_ct1.get_local_range(2);
  }
  item_ct1.barrier();

  reduceOne<THREADSPERBLOCK>(tid, s_scores, item_ct1);
  if (tid == 0)
  {
    SCORE[k] = s_scores[tid];
  }
}


static void calcNewScoreSortedAndSwap_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, FLOAT_TYPE *CTR2, int *INDEX, FLOAT_TYPE *SCORE, int *SEGOFFSET, int *MEDOID, int *RANDOM,
                                           sycl::nd_item<3> item_ct1,
                                           uint8_t *dpct_local)
{
  auto * array     = (float*)dpct_local;
  auto * s_scores  = (FLOAT_TYPE*)array;
  auto * s_center  = (FLOAT_TYPE*)&s_scores[item_ct1.get_local_range(2) + 1];
  auto * s_segment = (int*)&s_center[item_ct1.get_local_range(2)];
  auto * s_random  = (int*)&s_segment[2];

  int k = item_ct1.get_group(2);
  int tid = item_ct1.get_local_id(2);

  if (tid < 2)
  {
    s_segment[tid] = SEGOFFSET[k + tid];
  }
  if (tid == 0)
  {
    s_random[tid] = RANDOM[k];
  }
  if (tid == 0)
  {
    s_scores[item_ct1.get_local_range().get(2) + tid] = SCORE[k];
  }
  item_ct1.barrier();

  FLOAT_TYPE score = s_scores[item_ct1.get_local_range(2)];
  int endOffset = s_segment[1];

  s_scores[tid] = 0.0;
  unsigned int offsetN = s_segment[0];
  const auto iterationCount =
    (sycl::abs(endOffset - offsetN) + item_ct1.get_local_range().get(2) - 1) / item_ct1.get_local_range().get(2);
  int newMedoid = offsetN + s_random[0] % (endOffset - offsetN);

  offsetN += tid;
  for (auto idx{ 0U }; idx < iterationCount; ++idx)
  {
    const auto processElement = offsetN < endOffset;
    FLOAT_TYPE dist = 0.0;
    unsigned int offsetD = 0;

    while (offsetD < D)
    {
      if (offsetD + tid < D)
      {
        s_center[tid] = X[(offsetD + tid) * N + newMedoid];
        CTR2[k * D + offsetD + tid] = s_center[tid];
      }
      item_ct1.barrier();

      const auto minValue = sycl::min((unsigned int)(offsetD + item_ct1.get_local_range(2)), (unsigned int)D);
      for (unsigned int d = offsetD; d < minValue; d++)
      {
        if (processElement)
        {
          dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + offsetN));
        }
      }
      offsetD += item_ct1.get_local_range().get(2);
      item_ct1.barrier();
    }

    s_scores[tid] += distanceFinalizeGPU(1, &dist);
    offsetN += item_ct1.get_local_range().get(2);
  }
  item_ct1.barrier();

  reduceOne<THREADSPERBLOCK>(tid, s_scores, item_ct1);

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
      offsetD += item_ct1.get_local_range().get(2);
    }
  }
}


// ************ SECTION FOR PARALLEL DATA SORTING *******************  



template <unsigned int BLOCKSIZE>
static int parallelPrefixSum(int tid, int *DATA,
                             sycl::nd_item<3> item_ct1)
{
  unsigned int temp = 0;
  unsigned int sum = 0;
  unsigned int n = 2 * BLOCKSIZE;    // always work with 2 * tpb;

  for (unsigned idx{ 1024U }; idx > 1; idx /= 2)
  {
    const auto halfIdx{ idx / 2 };
    if (n >= idx) 
    {
      if (tid < halfIdx)
      {
        DATA[tid] += DATA[tid + halfIdx];
      }
      item_ct1.barrier();
    }
  }

  item_ct1.barrier();

  // broadcast and store reduced sum of elements in array across threads
  sum = DATA[0];
  item_ct1.barrier();
  if (tid == 0)
  {
    DATA[0] = 0;
  }

  for (unsigned idx{ 2U }; idx < 2048; idx *= 2)
  {
    const auto halfIdx{ idx / 2 };
    if (n >= idx) 
    {
      if (tid < halfIdx)
      {
        temp = DATA[tid];
        DATA[tid] += DATA[tid + halfIdx];
        DATA[tid + halfIdx] = temp;
      }
      item_ct1.barrier();
    }
  }

  return sum;
}



static void sort_getSegmentSize_CUDA(int N, int *ASSIGN, int *SEGSIZE,
                                     sycl::nd_item<3> item_ct1,
                                     uint8_t *dpct_local)
{
  auto * array = (float*)dpct_local;
  int * s_num  = (int*)array;

  unsigned int k   = item_ct1.get_group(2);
  unsigned int tid = item_ct1.get_local_id(2);
  unsigned int num = 0;

  unsigned int offsetN = tid;
  while (offsetN < N)
  {
    if (ASSIGN[offsetN] == k)
    {
      num++;
    }
    offsetN += THREADSPERBLOCK;
  }

  s_num[tid] = num;
  item_ct1.barrier();

  reduceOne<THREADSPERBLOCK>(tid, s_num, item_ct1);
  if (tid == 0)
  {
    SEGSIZE[k] = s_num[tid];
  }
}



static void sort_moveData_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *X2, int *INDEX, int *INDEX2, int *ASSIGN, int *SEGOFFSET,
                               sycl::nd_item<3> item_ct1,
                               uint8_t *dpct_local)
{
  auto * array     = (float*) dpct_local;
  auto * s_gather  = (int*)   array;
  auto * s_indices = (int*)  &s_gather[3 * THREADSPERBLOCK];
  auto * s_segment = (int*)  &s_indices[2 * THREADSPERBLOCK];

  bool scan1 = false;
  bool scan2 = false;

  int k                               = item_ct1.get_group(2);
  int tid                             = item_ct1.get_local_id(2);
  s_gather[tid]                       = 0;
  s_gather[tid + THREADSPERBLOCK]     = 0;
  s_gather[tid + 2 * THREADSPERBLOCK] = 0;
  if (tid < 2)
  {
    s_segment[tid] = SEGOFFSET[k + tid];
  }
  item_ct1.barrier();

  int bufferOffset    = s_segment[0] + tid;
  int bufferEndOffset = s_segment[1];
  int dataOffset      = tid;
  int windowSize      = 2 * THREADSPERBLOCK;
  int numFound        = 0;

  const auto iterationCount = sycl::max<unsigned>(
    (N + windowSize - 1) / windowSize, 
    (sycl::abs(bufferEndOffset - bufferOffset + tid) + THREADSPERBLOCK - 1) / THREADSPERBLOCK);

  for (auto idx{ 0U }; idx < iterationCount; ++idx)
  {
    const auto processElement = (dataOffset - tid) < N && (bufferOffset - tid + numFound) < bufferEndOffset;

    if (processElement)
    {
      scan1 = ((dataOffset < N) && (ASSIGN[dataOffset] == k));
      scan2 = ((dataOffset + THREADSPERBLOCK < N) && (ASSIGN[dataOffset + THREADSPERBLOCK] == k));
      s_indices[tid] = 0;
      s_indices[tid + THREADSPERBLOCK] = 0;
      if (scan1)
      {
        s_indices[tid] = 1;
      }
      if (scan2)
      {
        s_indices[tid + THREADSPERBLOCK] = 1;
      }

      int nonZero = parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices, item_ct1);
      if (scan1)
      {
        s_gather[numFound + s_indices[tid]] = dataOffset;
      }
      if (scan2)
      {
        s_gather[numFound + s_indices[tid + THREADSPERBLOCK]] = dataOffset + THREADSPERBLOCK;
      }
      numFound += nonZero;
    }
    item_ct1.barrier();

    while (numFound >= THREADSPERBLOCK && processElement)
    {
      for (unsigned int d = 0; d < D; d++)
      {
        X2[bufferOffset + d * N] = X[s_gather[tid] + d * N];
      }

      INDEX2[bufferOffset]                = INDEX[s_gather[tid]];
      numFound                           -= THREADSPERBLOCK;
      s_gather[tid]                       = s_gather[tid + THREADSPERBLOCK];
      s_gather[tid + THREADSPERBLOCK]     = s_gather[tid + 2 * THREADSPERBLOCK];
      s_gather[tid + 2 * THREADSPERBLOCK] = 0;
      bufferOffset                       += THREADSPERBLOCK;
    }
    dataOffset += windowSize;
    item_ct1.barrier();
  }
  
  if (bufferOffset < bufferEndOffset)
  {
    // for each dimension, transfer data
    for (unsigned int d = 0; d < D; d++)
    {
      X2[bufferOffset + d * N] = X[s_gather[tid] + d * N];
    }

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
    dpct::get_default_queue()
        .memcpy(intermediates, INPUT, sizeof(int) * N)
        .wait();
    // value at location 0 is okay
    for (unsigned int i = 1; i < N; i++)
    { 
        intermediates[i] += intermediates[i - 1];
    }
    // transfer results to device
    dpct::get_default_queue()
        .memcpy(OUTPUT, intermediates, sizeof(int) * N)
        .wait();
    free(intermediates);
}



void sortData(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *X2, int *INDEX, int *INDEX2, int *ASSIGN, int *SEGSIZE, int *SEGOFFSET)
{
    sycl::range<3> block(THREADSPERBLOCK, 1,
                         1);       // set number of threads per block
    sycl::range<3> gridK(K, 1, 1); // K blocks of threads in grid

    // loop over all data points, detect those that are assigned to cluster k,
    // determine unique memory indices in contiguous stretch of shared memory 
    // using string compaction with parallel prefix sum, then move data to buffer
    
    // run over all data points and detect those assigned to cluster K
    int sMem = (sizeof(int) * THREADSPERBLOCK);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(sMem), cgh);

        auto dpct_global_range = gridK * block;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block.get(2), block.get(1), block.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                sort_getSegmentSize_CUDA(N, ASSIGN, SEGSIZE, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
            });
    });

    // first segment offset is 0 (as of initialization), compute the others as running sum over segment sizes
    serialPrefixSum_KMDCUDA(K, SEGSIZE, SEGOFFSET + 1);
    
    // now move the data from X to X2
    sMem = sizeof(int)*(5*THREADSPERBLOCK + 2);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(sMem), cgh);

        auto dpct_global_range = gridK * block;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block.get(2), block.get(1), block.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                sort_moveData_CUDA(N, D, X, X2, INDEX, INDEX2, ASSIGN,
                                   SEGOFFSET, item_ct1,
                                   dpct_local_acc_ct1.get_pointer());
            });
    });
}



FLOAT_TYPE kmedoidsGPU(int N, int K, int D, FLOAT_TYPE *x, int *medoid, int *assign, unsigned int maxIter, DataIO *data)
{
    // CUDA kernel parameters
    sycl::range<3> block(THREADSPERBLOCK, 1, 1);
    sycl::range<3> gridK(K, 1, 1);
    sycl::range<3> gridN((int)ceil((FLOAT_TYPE)N / (FLOAT_TYPE)THREADSPERBLOCK),
                         1, 1);
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
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                dpct_local_acc_ct1(sycl::range<1>(sMemAssign), cgh);

            auto dpct_global_range = gridN * block;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(dpct_global_range.get(2),
                                   dpct_global_range.get(1),
                                   dpct_global_range.get(0)),
                    sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    assignToClusters_KMDCUDA(N, K, D, x_d, ctr_d, assign_d,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                });
        });

        // sort by assignment (O(K*D*N))
        sortData(N, K, D, x_d, x2_d, indices_d, indices2_d, assign_d, segsize_d, segoffs_d);
        
        // swap sorted buffer with unsorted data
        FLOAT_TYPE *dataTemp_d = x_d; x_d = x2_d; x2_d = dataTemp_d;
        
        // swap shuffled indices with those from previous iteration
        int *indTemp_d = indices_d; indices_d = indices2_d; indices2_d = indTemp_d;
        
        // get score per cluster for sorted data
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                dpct_local_acc_ct1(sycl::range<1>(sMemScore), cgh);

            auto dpct_global_range = gridK * block;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(dpct_global_range.get(2),
                                   dpct_global_range.get(1),
                                   dpct_global_range.get(0)),
                    sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    calcScoreSorted_CUDA(N, D, x_d, ctr_d, s_d, segoffs_d,
                                         item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                });
        });

        // generate K random numbers, do this on CPU
        for (unsigned int k = 0; k < K; k++) random[k] = rand();
        dpct::get_default_queue()
            .memcpy(random_d, random, sizeof(int) * K)
            .wait();

        // compute score for randomly selected new set of medoids and swap medoids score improves
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                dpct_local_acc_ct1(sycl::range<1>(sMemScoreSwap), cgh);

            auto dpct_global_range = gridK * block;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(dpct_global_range.get(2),
                                   dpct_global_range.get(1),
                                   dpct_global_range.get(0)),
                    sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    calcNewScoreSortedAndSwap_CUDA(
                        N, D, x_d, ctr_d, ctr2_d, indices_d, s_d, segoffs_d,
                        medoid_d, random_d, item_ct1,
                        dpct_local_acc_ct1.get_pointer());
                });
        });

        iter++;
    } 
    
    // copy scores per cluster back to the host and do reduction on CPU
    dpct::get_default_queue().memcpy(s, s_d, sizeof(FLOAT_TYPE) * K).wait();
    FLOAT_TYPE score = 0.0;
    for (int i = 0; i < K; i++) score += s[i];
    
    // copy medoids back to host
    sycl::queue &q_ct0 = dpct::get_default_queue();
    q_ct0.wait();
    q_ct0.memcpy(medoid, medoid_d, sizeof(int) * K).wait();
    // copy medoid coordinates back to host
    q_ct0.memcpy(ctr, ctr_d, sizeof(FLOAT_TYPE) * K * D).wait();
    // copy assignments back to host
    q_ct0.memcpy(assign, assign_d, sizeof(int) * N).wait();

    // free memory
    sycl::free(x_d, dpct::get_default_context());
    sycl::free(x2_d, dpct::get_default_context());
    sycl::free(ctr_d, dpct::get_default_context());
    sycl::free(ctr2_d, dpct::get_default_context());
    sycl::free(random_d, dpct::get_default_context());
    sycl::free(segsize_d, dpct::get_default_context());
    sycl::free(segoffs_d, dpct::get_default_context());
    sycl::free(assign_d, dpct::get_default_context());
    sycl::free(s_d, dpct::get_default_context());
    sycl::free(medoid_d, dpct::get_default_context());
    free(ctr);
    free(random);
    free(s);
    
    return score;
}

