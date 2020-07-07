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
 * \File kmeans_gpu.cu
 * \brief A basic CUDA K-means implementation
 *
 * A module of the CAMPAIGN data clustering library for parallel architectures
 * Implements parallel K-means clustering (base implementation) on the GPU
 *
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 12/2/2010
 * \version 1.0
 **/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kmeans_gpu.h"
#include <cmath>

using namespace std;

template <unsigned int BLOCKSIZE, class T>
static void reduceOne(int tid, T *s_A,
                      sycl::nd_item<3> item_ct1)
{
  for (unsigned idx{ 1024u }; idx > 1u; idx /= 2u)
  {
    if (BLOCKSIZE >= idx)
    {
      const auto halfIdx = idx / 2;
      if (tid < halfIdx)
      {
        s_A[tid] += s_A[tid + halfIdx];
      }
      item_ct1.barrier();
    }
  }
}


template <unsigned int BLOCKSIZE, class T, class U>
static void reduceTwo(int tid, T *s_A, U *s_B,
                      sycl::nd_item<3> item_ct1)
{
  for (unsigned idx{ 1024u }; idx > 1u; idx /= 2u)
  {
    if (BLOCKSIZE >= idx)
    {
      const auto halfIdx = idx / 2;
      if (tid < halfIdx)
      {
        s_A[tid] += s_A[tid + halfIdx];
        s_B[tid] += s_B[tid + halfIdx];
      }
      item_ct1.barrier();
    }
  }
}


static void assignToClusters_KMCUDA(int N, int K, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN,
                                    sycl::nd_item<3> item_ct1,
                                    uint8_t *dpct_local)
{
  auto * array     = (float*)dpct_local;
  auto * s_center  = (FLOAT_TYPE*)array;

  unsigned int t   = item_ct1.get_global_id(2);
  unsigned int tid = item_ct1.get_local_id(2);

  const bool processElement = t < N;
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
    dist = distanceFinalizeGPU(1, &dist);

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


static void calcScore_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN, FLOAT_TYPE *SCORE,
                           sycl::nd_item<3> item_ct1,
                           uint8_t *dpct_local)
{
  auto * array    = (float*)dpct_local;
  auto * s_scores = (FLOAT_TYPE*)array;
  auto * s_center = (FLOAT_TYPE*)&s_scores[item_ct1.get_local_range(2)];

  int k           = item_ct1.get_group(2);
  int tid         = item_ct1.get_local_id(2);

  s_scores[tid] = 0.0;
  s_center[tid] = 0.0;
  item_ct1.barrier();

  unsigned int offsetN = tid;
  unsigned iterationCount = (N + item_ct1.get_local_range(2) - 1) / item_ct1.get_local_range(2);
  for (unsigned idx{ 0U }; idx <= iterationCount; ++idx)
  {
    FLOAT_TYPE dist = 0.0;
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
      // at each iteration read up to tpb centroid components from global mem (coalesced)
      if (offsetD + tid < D)
      {
        s_center[tid] = CTR[k * D + offsetD + tid];
      }
      item_ct1.barrier();

      if ((offsetN < N) && (ASSIGN[offsetN] == k))
      {
        const auto minValue = sycl::min((unsigned int)(offsetD + item_ct1.get_local_range(2)), (unsigned int)D);
        for (unsigned int d = offsetD; d < minValue; d++)
        {
          dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + offsetN));
        }
      }
      offsetD += item_ct1.get_local_range().get(2);
      item_ct1.barrier();
    }
    // update partial score
    s_scores[tid] += distanceFinalizeGPU(1, &dist);
    item_ct1.barrier();
    offsetN += item_ct1.get_local_range().get(2);
  }
  item_ct1.barrier();

  reduceOne<THREADSPERBLOCK>(tid, s_scores, item_ct1);
  item_ct1.barrier();
  if (tid == 0)
  {
    SCORE[k] = s_scores[tid];
  }
}



static void calcCentroids_CUDA(int N, int D, FLOAT_TYPE *X, FLOAT_TYPE *CTR, int *ASSIGN,
                               sycl::nd_item<3> item_ct1,
                               uint8_t *dpct_local)
{
  auto * array         = (float*)dpct_local;
  auto * s_numElements = (int*)array;
  auto * s_centerParts = (FLOAT_TYPE*)&s_numElements[item_ct1.get_local_range(2)];

  int k                = item_ct1.get_group(2);
  int tid              = item_ct1.get_local_id(2);

  FLOAT_TYPE clusterSize = 0.0;

  s_numElements[tid] = 0;
  item_ct1.barrier();

  for (unsigned int d = 0; d < D; d++)
  {
    s_centerParts[tid] = 0.0;
    item_ct1.barrier();

    unsigned int offset = tid;
    unsigned iterationCount = (N + item_ct1.get_local_range(2) - 1) / item_ct1.get_local_range(2);
    for (unsigned idx{ 0U }; idx <= iterationCount; ++idx)
    {
      if ((offset < N) && (ASSIGN[offset] == k))
      {
        s_centerParts[tid] += X[d * N + offset];
        if (d == 0)
        {
          s_numElements[tid]++;
        }
      }
      item_ct1.barrier();
      offset += item_ct1.get_local_range(2);
    }
    item_ct1.barrier();

    if (d == 0)
    {
      reduceTwo<THREADSPERBLOCK>(tid, s_centerParts, s_numElements, item_ct1);
      if (tid == 0)
      {
        clusterSize = (FLOAT_TYPE)s_numElements[tid];
      }
    }
    else
    {
      reduceOne<THREADSPERBLOCK>(tid, s_centerParts, item_ct1);
    }

    if ((tid == 0) && (clusterSize > 0))
    {
      CTR[k * D + d] = s_centerParts[tid] / clusterSize;
    }
    item_ct1.barrier();
  }
}


FLOAT_TYPE kmeansGPU(int N, int K, int D, FLOAT_TYPE *x, FLOAT_TYPE *ctr, int *assign, unsigned int maxIter, DataIO *data)
{
  dpct::device_ext & dev_ct1 = dpct::get_current_device();
  sycl::queue & q_ct1        = dev_ct1.default_queue();

  sycl::range<3> block(THREADSPERBLOCK, 1, 1);
  sycl::range<3> gridK(K, 1, 1);
  sycl::range<3> gridN((int)ceil((FLOAT_TYPE)N / (FLOAT_TYPE)THREADSPERBLOCK), 1, 1);

  int sMemAssign  = (sizeof(FLOAT_TYPE) * THREADSPERBLOCK);
  int sMemScore   = (sizeof(FLOAT_TYPE) * 2 * THREADSPERBLOCK);
  int sMemCenters = (sizeof(FLOAT_TYPE) * THREADSPERBLOCK + sizeof(int) * THREADSPERBLOCK);

  auto * x_d       = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N * D, x);
  auto * ctr_d     = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * K * D, (void*)ctr);
  auto * assign_d  = (int*)       data->allocDeviceMemory(sizeof(int) * N);
  auto * s_d       = (FLOAT_TYPE*)data->allocZeroedDeviceMemory(sizeof(FLOAT_TYPE) * K);

  auto * s         = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * K);

  FLOAT_TYPE oldscore = -1000.0, score = 0.0;
  if (maxIter < 1)
  {
    maxIter = INT_MAX;
  }
  unsigned int iter = 0;

  while (iter < maxIter && ((score - oldscore) * (score - oldscore)) > EPS)
  {
    oldscore = score;

    using LocalReadWriteAccessor = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write, sycl::access::target::local>;
    if (iter > 0)
    {
      q_ct1.submit([&](sycl::handler& cgh) {
        LocalReadWriteAccessor dpct_local_acc_ct1(sycl::range<1>(sMemCenters), cgh);
        auto dpct_global_range = gridK * block;
        cgh.parallel_for(sycl::nd_range<3>(
          sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
          sycl::range<3>(block.get(2), block.get(1), block.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
          calcCentroids_CUDA(N, D, x_d, ctr_d, assign_d, item_ct1,
            dpct_local_acc_ct1.get_pointer());
        });
      }).wait();
    }
    iter++;

    q_ct1.submit([&](sycl::handler& cgh) 
    {
      LocalReadWriteAccessor dpct_local_acc_ct1(sycl::range<1>(sMemAssign), cgh);
      auto dpct_global_range = gridN * block;

      cgh.parallel_for(
        sycl::nd_range<3>(
          sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
          sycl::range<3>(block.get(2), block.get(1), block.get(0))),
        [=](sycl::nd_item<3> item_ct1) 
      {
          assignToClusters_KMCUDA(N, K, D, x_d, ctr_d, assign_d, item_ct1, dpct_local_acc_ct1.get_pointer());
      });
    }).wait();

    // get score per cluster for unsorted data
    q_ct1.submit([&](sycl::handler& cgh) {
      LocalReadWriteAccessor dpct_local_acc_ct1(sycl::range<1>(sMemScore), cgh);

      auto dpct_global_range = gridK * block;

      cgh.parallel_for(
        sycl::nd_range<3>(
          sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)),
          sycl::range<3>(block.get(2), block.get(1), block.get(0))),
        [=](sycl::nd_item<3> item_ct1) 
      {
        calcScore_CUDA(N, D, x_d, ctr_d, assign_d, s_d, item_ct1, dpct_local_acc_ct1.get_pointer());
      });
    }).wait();

    // copy scores per cluster back to the host and do reduction on CPU
    q_ct1.memcpy(s, s_d, sizeof(FLOAT_TYPE) * K).wait();
    score = 0.0;
    for (int i = 0; i < K; i++) score += s[i];
  }
  cout << "Number of iterations: " << iter << endl;

  q_ct1.wait();
  q_ct1.memcpy(ctr, ctr_d, sizeof(FLOAT_TYPE) * K * D).wait();
  q_ct1.memcpy(assign, assign_d, sizeof(int) * N).wait();

  sycl::free(x_d, dpct::get_default_context());
  sycl::free(ctr_d, dpct::get_default_context());
  sycl::free(assign_d, dpct::get_default_context());
  sycl::free(s_d, dpct::get_default_context());
  free(s);

  return score;
}

