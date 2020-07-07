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
  * \file kcenters_gpu.cu
  * \brief A CUDA K-centers implementation
  *
  * Implements parallel K-centers clustering on the GPU
  *
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 12/2/2010
  * \version 1.0
  **/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kcenters_gpu.h"
#include <cmath>

using namespace std;

template <unsigned int BLOCKSIZE, class T, class U>
static void parallelMax(int tid, T* s_A, U* s_B, sycl::nd_item<3> item_ct1)
{
  const auto gIdx = item_ct1.get_global_id(0);
  const auto gSize = item_ct1.get_global_range(0);
  for (unsigned idx{ 1024u }; idx > 1u; idx /= 2u)
  {
    if (BLOCKSIZE >= idx)
    {
      const auto halfIdx = idx / 2;
      if (tid < halfIdx && gIdx + halfIdx < gSize && s_A[tid + halfIdx] > s_A[tid])
      {
        s_A[tid] = s_A[tid + halfIdx];
        s_B[tid] = s_B[tid + halfIdx];
      }
      item_ct1.barrier();
    }
  }
}

void checkCentroid_sycl(int N, int D, int iter, int centroid, FLOAT_TYPE* X,
  FLOAT_TYPE* CTR, FLOAT_TYPE* DIST, int* ASSIGN,
  FLOAT_TYPE* MAXDIST, int* MAXID,
  sycl::nd_item<3> item_ct1, uint8_t* dpct_local)
{
  auto * array     = (float*)      dpct_local;
  auto * s_dist    = (FLOAT_TYPE*) array;
  auto * s_ID      = (int*)        &s_dist[item_ct1.get_local_range(0)];
  auto * s_ctr     = (FLOAT_TYPE*) &s_ID[item_ct1.get_local_range(0)];

  unsigned int tid = item_ct1.get_local_id(0);
  unsigned int t   = item_ct1.get_global_id(0);

  s_dist[tid]      = 0.0;
  s_ID[tid]        = t;
  item_ct1.barrier();

  const bool processElement = t < N;
  FLOAT_TYPE dist = 0.0;
  int   offsetD = 0;

  while (offsetD < D)
  {
    if (offsetD + tid < D)
    {
      s_ctr[tid] = CTR[offsetD + tid];
    }
    item_ct1.barrier();

    const auto minValue = sycl::min((unsigned int)item_ct1.get_local_range(0), (unsigned int)(D - offsetD));
    for (int d = 0; d < minValue; d++)
    {
      if (processElement)
      {
        dist += distanceComponentGPU(s_ctr + d - offsetD, X + (offsetD + d) * N + t);
      }
    }
    offsetD += item_ct1.get_local_range(0);
    item_ct1.barrier();
  }
  dist = distanceFinalizeGPU<FLOAT_TYPE>(1, &dist);

  if (processElement)
  {
    FLOAT_TYPE currDist = DIST[t];
    if (dist < currDist)
    {
      DIST[t] = currDist = dist;
      ASSIGN[t] = iter;
    }
    s_dist[tid] = currDist;
  }
  item_ct1.barrier();

  parallelMax<THREADSPERBLOCK>(tid, s_dist, s_ID, item_ct1);
  item_ct1.barrier();
  if (tid == 0)
  {
    MAXDIST[item_ct1.get_group(0)] = s_dist[tid];
    MAXID[item_ct1.get_group(0)] = s_ID[tid];
  }
  item_ct1.barrier();
}



void kcentersGPU(int N, int K, int D, FLOAT_TYPE* x, int* assign, FLOAT_TYPE* dist, int* centroids, int seed, DataIO* data)
{
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue&      q_ct1   = dev_ct1.default_queue();

  int numBlock = (int)ceil((FLOAT_TYPE)N / (FLOAT_TYPE)THREADSPERBLOCK);
  sycl::range<3> block(THREADSPERBLOCK, 1, 1);
  sycl::range<3> grid(numBlock, 1, 1);
  int sMem = (2 * sizeof(FLOAT_TYPE) + sizeof(int)) * THREADSPERBLOCK;

  auto dist_d         = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N,       (void*)dist);
  auto assign_d       = (int*)       data->allocDeviceMemory(sizeof(int)        * N);
  auto x_d            = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * N * D,   (void*)x);
  auto ctr_d          = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * D);
  auto maxDistBlock_d = (FLOAT_TYPE*)data->allocDeviceMemory(sizeof(FLOAT_TYPE) * numBlock);
  auto maxIdBlock_d   = (int*)       data->allocDeviceMemory(sizeof(int)        * numBlock);

  auto maxDistBlock   = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numBlock);
  auto maxID          = (int*)       malloc(sizeof(int));
  auto ctr            = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * D);

  int centroid = seed;
  for (int k = 0; k < K; ++k)
  {
    for (int d = 0; d < D; d++)
    {
      ctr[d] = x[d * N + centroid];
    }
    q_ct1.memcpy(ctr_d, ctr, sizeof(FLOAT_TYPE) * D).wait();
    centroids[k] = centroid;

    using LocalRwAccessor = sycl::accessor<uint8_t, 1, sycl::access::mode::read_write, sycl::access::target::local>;
    q_ct1.submit([&](sycl::handler& cgh) 
    {
      LocalRwAccessor dpct_local_acc_ct1(sycl::range<1>(sMem), cgh);
      auto dpct_global_range = grid * block;

      cgh.parallel_for(sycl::nd_range<3>(dpct_global_range, block),
        [=](sycl::nd_item<3> item_ct1) 
      {
        checkCentroid_sycl(N, D, k, centroid, x_d, ctr_d, dist_d, assign_d, maxDistBlock_d, maxIdBlock_d,
                           item_ct1, dpct_local_acc_ct1.get_pointer());
      });
    }).wait();

    q_ct1.memcpy(maxDistBlock, maxDistBlock_d, sizeof(FLOAT_TYPE) * numBlock).wait();
    int tempMax = 0;
    for (int i = 1; i < numBlock; i++)
    {
      if (maxDistBlock[i] > maxDistBlock[tempMax])
      {
        tempMax = i;
      }
    }
    q_ct1.memcpy(maxID, maxIdBlock_d + tempMax, sizeof(int)).wait();
    centroid = maxID[0];
  }
  // copy final assignments back to host
  q_ct1.memcpy(assign, assign_d, sizeof(int) * N).wait();

  // free up memory
  sycl::free(dist_d, q_ct1);
  sycl::free(assign_d, q_ct1);
  sycl::free(x_d, q_ct1);
  sycl::free(ctr_d, q_ct1);
  sycl::free(maxDistBlock_d, q_ct1);
  sycl::free(maxIdBlock_d, q_ct1);
  free(maxDistBlock);
  free(maxID);
  free(ctr);
}


