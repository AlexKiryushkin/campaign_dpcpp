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
 * \file hierarchical_gpu.cu
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
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "hierarchical_gpu.h"
#include <cmath>

#endif

using namespace std;

template <unsigned int BLOCKSIZE, class T, class U>
static void reduceMinTwo(int tid, T *s_A, U *s_B,
                         sycl::nd_item<3> item_ct1)
{
  for (unsigned idx{ 1024u }; idx > 1; idx /= 2)
  {
    const unsigned halfIdx = idx / 2;
    if (BLOCKSIZE >= idx)
    {
      if (tid < halfIdx)
      {
        if (s_A[tid + halfIdx] == s_A[tid])
        {
          s_B[tid] = sycl::min(s_B[tid], s_B[tid + halfIdx]);
        }
        if (s_A[tid + halfIdx] < s_A[tid])
        {
          s_A[tid] = s_A[tid + halfIdx];
          s_B[tid] = s_B[tid + halfIdx];
        }
      }
      item_ct1.barrier();
    }
  }
}


static void calcDistanceMatrix_CUDA(int N, int D, FLOAT_TYPE *X, int *NEIGHBOR, FLOAT_TYPE *NEIGHBORDIST,
                                    sycl::nd_item<3> item_ct1)
{
  int t = item_ct1.get_global_id(2);
  if (t < sycl::ceil((float)((FLOAT_TYPE)N / 2.0)))
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


void min_CUDA(unsigned int N, unsigned int iter, FLOAT_TYPE *INPUT, int *INKEY, FLOAT_TYPE *OUTPUT, int *OUTKEY,
              sycl::nd_item<3> item_ct1,
              uint8_t *dpct_local)
{
    auto * array   = (float *)dpct_local;
    auto * s_value = (FLOAT_TYPE*) array;
    auto * s_key   = (int *)&s_value[item_ct1.get_local_range(2)];

    unsigned int tid = item_ct1.get_local_id(2);
    unsigned int t = item_ct1.get_global_id(2);
    s_value[tid] = FLT_MAX;
    s_key  [tid] = 0;
    
    if (t < N)
    {
        s_value[tid] = INPUT[t];
        s_key  [tid] = (iter == 0) ? t : INKEY[t];
    }
    item_ct1.barrier();

    reduceMinTwo<THREADSPERBLOCK>(tid, s_value, s_key, item_ct1);
    if (tid == 0)
    {
        OUTPUT[item_ct1.get_group(2)] = s_value[tid];
        OUTKEY[item_ct1.get_group(2)] = s_key[tid];
    }
}



int getMinDistIndex(int numClust, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES)
{
    // prepare CUDA parameters
    sycl::range<3> block(THREADSPERBLOCK, 1, 1);
    int numBlocks =
        (int)ceil((FLOAT_TYPE)numClust / (FLOAT_TYPE)THREADSPERBLOCK);
    sycl::range<3> gridN(numBlocks, 1, 1);
    int sMem = (sizeof(FLOAT_TYPE) + sizeof(int)) * THREADSPERBLOCK;
    
    // cooperatively reduce distance array to find block-wise minimum and index
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(sMem), cgh);

        auto dpct_global_range = gridN * block;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block.get(2), block.get(1), block.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                min_CUDA(numClust, 0, DISTS, INDICES, REDUCED_DISTS,
                         REDUCED_INDICES, item_ct1,
                         dpct_local_acc_ct1.get_pointer());
            });
    });

    // repeat reduction until single value and key pair left
    while (numBlocks > 1)
    {
        int nElements = numBlocks;
        numBlocks =
            (int)ceil((FLOAT_TYPE)nElements / (FLOAT_TYPE)THREADSPERBLOCK);
        sycl::range<3> nBlocks(numBlocks, 1, 1);

        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                dpct_local_acc_ct1(sycl::range<1>(sMem), cgh);

            auto dpct_global_range = nBlocks * block;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(dpct_global_range.get(2),
                                   dpct_global_range.get(1),
                                   dpct_global_range.get(0)),
                    sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    min_CUDA(nElements, 1, REDUCED_DISTS, REDUCED_INDICES,
                             REDUCED_DISTS, REDUCED_INDICES, item_ct1,
                             dpct_local_acc_ct1.get_pointer());
                });
        });
    }
    
    // copy result, i.e. index of element with minimal distance, to host
    int *ind = (int*) malloc(sizeof(int));
    dpct::get_default_queue()
        .memcpy(ind, REDUCED_INDICES, sizeof(int) * 1)
        .wait();
    int index = ind[0];
    
    // free memory
    free(ind);
    
    return index;
}


static void mergeElementsInsertAtA_CUDA(int N, int D, int indexA, int indexB, FLOAT_TYPE *X,
                                        sycl::nd_item<3> item_ct1)
{
    // compute global thread number
    int d = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
    if (d < D)
    {
        X[d * N + indexA] = (X[d * N + indexA] + X[d * N + indexB]) / 2.0;
    }
}



static void computeAllDistancesToA_CUDA(int N, int numClust, int D, int indexA, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES,
                                        sycl::nd_item<3> item_ct1,
                                        uint8_t *dpct_local) // includes distance from old B to A, which will later be discarded
{
    int numThreads =
        item_ct1.get_local_range().get(2); // number of threads in block

    // define shared mem for first part of reduction step
    auto array = (float *)dpct_local; // declare variable in shared memory
    FLOAT_TYPE* s_dist  = (FLOAT_TYPE*) array;                  // dynamically allocate FLOAT_TYPE array at offset 0 to hold intermediate distances
    int*   s_index = (int*  ) &s_dist[numThreads];    // dynamically allocate int array at offset blockDim.x to hold intermediate indices
    FLOAT_TYPE* s_posA  = (FLOAT_TYPE*) &s_index[numThreads];   // dynamically allocate memory for position of element A

    int t = numThreads * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);   // global thread ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID

    if (t < numClust) s_dist[tid] = 0.0;
    
    // for each dimension
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb components of element at j into shared memory (non-coalesced global memory access)
        if (offsetD + tid < D) s_posA[tid] = X[(offsetD + tid) * N + indexA];
        item_ct1.barrier();
        // compute distances between up to tpb clusters and cluster at j
        for (unsigned int d = offsetD;
             d <
             sycl::min((unsigned int)(offsetD + numThreads), (unsigned int)D);
             d++)
        {
            if (t < numClust) s_dist[tid] += distanceComponentGPU(X + d * N + t, s_posA + d - offsetD);
        }
        offsetD += numThreads;
        item_ct1.barrier();
    }
    if (t < numClust) s_dist[tid] = distanceFinalizeGPU(1, s_dist + tid);
    s_index[tid] = t;
    item_ct1.barrier();

    // for clusters in sequence after A
    if (t > indexA && t < numClust)
    {
        FLOAT_TYPE dist = DISTS[t];
        if (s_dist[tid] == dist)
            INDICES[t] = sycl::min(indexA, INDICES[t]);
        else if (s_dist[tid] < dist)
        {
            DISTS  [t] = s_dist[tid];
            INDICES[t] = indexA;
        }
    }
    if (t >= indexA) s_dist[tid] = FLT_MAX;
    item_ct1.barrier();

    // find minimum distance in array and index of corresponding cluster
    reduceMinTwo<THREADSPERBLOCK>(tid, s_dist, s_index, item_ct1);

    // write result for this block to global mem
    if (tid == 0)
    {
        REDUCED_DISTS[item_ct1.get_group(2)] = s_dist[tid];
        REDUCED_INDICES[item_ct1.get_group(2)] = s_index[tid];
    }
}


static void updateElementA_CUDA(int indexA, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES)
{
    DISTS  [indexA] = REDUCED_DISTS  [0];
    INDICES[indexA] = REDUCED_INDICES[0];
}



void updateDistanceAndIndexForCluster(int indexA, int sMem, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES) // set smallest dist from A
{
    sycl::range<3> block(THREADSPERBLOCK, 1, 1);
    // after pre-reduction there will now be ceil((indexA - 1) / THREADSPERBLOCK) distances and indices in the arrays
    int numBlocks =
        (int)ceil((FLOAT_TYPE)(indexA - 1) / (FLOAT_TYPE)THREADSPERBLOCK);
    // finish reduction on GPU
    while (numBlocks > 1)
    {
        sycl::range<3> nBlocks(
            (int)ceil((FLOAT_TYPE)numBlocks / (FLOAT_TYPE)THREADSPERBLOCK), 1,
            1);

        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                dpct_local_acc_ct1(sycl::range<1>(sMem), cgh);

            auto dpct_global_range = nBlocks * block;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(dpct_global_range.get(2),
                                   dpct_global_range.get(1),
                                   dpct_global_range.get(0)),
                    sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    min_CUDA(numBlocks, 1, REDUCED_DISTS, REDUCED_INDICES,
                             REDUCED_DISTS, REDUCED_INDICES, item_ct1,
                             dpct_local_acc_ct1.get_pointer());
                });
        });

        numBlocks = nBlocks[0];
    }
    // update min distance and index for element A
    sycl::range<3> numB(1, 1, 1);

    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = numB * numB;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(numB.get(2), numB.get(1), numB.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                updateElementA_CUDA(indexA, DISTS, INDICES, REDUCED_DISTS,
                                    REDUCED_INDICES);
            });
    });
}



static void moveCluster_CUDA(int N, int D, int indexB, int indexN, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES,
                             sycl::nd_item<3> item_ct1)
{
    // compute global thread number
    int d = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);
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


static void computeDistancesToBForPLargerThanB_CUDA(int N, int D, int indexB, int numElements, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES,
                                                    sycl::nd_item<3> item_ct1,
                                                    uint8_t *dpct_local)
{
    int numThreads =
        item_ct1.get_local_range().get(2); // number of threads in block

    // define shared mem for first part of reduction step
    auto array = (float *)dpct_local; // declare variable in shared memory
    FLOAT_TYPE* s_posB = (FLOAT_TYPE*) array;                  // dynamically allocate memory for position of element B

    int t = numThreads * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);   // global thread ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID

    FLOAT_TYPE dist = 0.0;
    
    // for each dimension
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb components of element at j into shared memory (non-coalesced global memory access)
        if (offsetD + tid < D) s_posB[tid] = X[(offsetD + tid) * N + indexB];
        item_ct1.barrier();
        // compute distances between up to tpb clusters and cluster at j
        for (unsigned int d = offsetD;
             d <
             sycl::min((unsigned int)(offsetD + numThreads), (unsigned int)D);
             d++)
        {
            if (t < numElements) dist += distanceComponentGPU(X + d * N + (indexB + 1) + t, s_posB + d - offsetD);
        }
        offsetD += numThreads;
        item_ct1.barrier();
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



static void recomputeMinDistanceForElementAt_j_CUDA(int N, int D, int indexJ, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES,
                                                    sycl::nd_item<3> item_ct1,
                                                    uint8_t *dpct_local)
{
    int numThreads =
        item_ct1.get_local_range().get(2); // number of threads in block

    // allocate shared memory, requires 2*tpb*sizeof(FLOAT_TYPE) + tpb*sizeof(int)
    auto array = (float *)dpct_local; // declare variable in shared memory
    FLOAT_TYPE* s_dist  = (FLOAT_TYPE*) array;                // dynamically allocate FLOAT_TYPE array at offset 0 to hold intermediate distances
    int*   s_index = (int*  ) &s_dist[numThreads];  // dynamically allocate int array at offset blockDim.x to hold intermediate indices
    FLOAT_TYPE* s_posJ  = (FLOAT_TYPE*) &s_index[numThreads]; // dynamically allocate memory for tpb components of element at j

    int t = numThreads * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);   // global thread ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID

    s_dist[tid] = FLT_MAX;
    if (t < indexJ) s_dist[tid] = 0.0;
    
    // for each dimension
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb components of element at j into shared memory (non-coalesced global memory access)
        if (offsetD + tid < D) s_posJ[tid] = X[(offsetD + tid) * N + indexJ];
        item_ct1.barrier();
        // compute distances between up to tpb clusters and cluster at j
        for (unsigned int d = offsetD;
             d <
             sycl::min((unsigned int)(offsetD + numThreads), (unsigned int)D);
             d++)
        {
            if (t < indexJ) s_dist[tid] += distanceComponentGPU(X + d * N + t, s_posJ + d - offsetD);
        }
        offsetD += numThreads;
        item_ct1.barrier();
    }
    if (t < indexJ) 
    {
        s_dist[tid] = distanceFinalizeGPU(1, s_dist + tid);
        s_index[tid] = t;
    }
    item_ct1.barrier();
    // find minimum distance in array and index of corresponding cluster
    reduceMinTwo<THREADSPERBLOCK>(tid, s_dist, s_index, item_ct1);

    // write result for this block to global mem
    if (tid == 0)
    {
        REDUCED_DISTS[item_ct1.get_group(2)] = s_dist[tid];
        REDUCED_INDICES[item_ct1.get_group(2)] = s_index[tid];
    }  
}



int* hierarchicalGPU(int N, int D, FLOAT_TYPE *x, DataIO *data)
{
    // CUDA kernel parameters
    sycl::range<3> block(THREADSPERBLOCK, 1, 1);
    int numBlocks = (int)ceil((FLOAT_TYPE)N / (FLOAT_TYPE)THREADSPERBLOCK);
    int numBlocksD = (int)ceil((FLOAT_TYPE)D / (FLOAT_TYPE)THREADSPERBLOCK);
    sycl::range<3> gridN(numBlocks, 1, 1);
    sycl::range<3> gridD(numBlocksD, 1, 1);
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
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        auto dpct_global_range = gridN * block;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(block.get(2), block.get(1), block.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                calcDistanceMatrix_CUDA(N, D, x_d, closestClust_d,
                                        closestDist_d, item_ct1);
            });
    });

    // copy closest cluster indices back to host
    dpct::get_default_queue()
        .memcpy(closestClust, closestClust_d, sizeof(int) * N)
        .wait();

    // initialize clustID
    for (int i = 0; i < N; i++) clustID[i] = i;
    
    last = N;
    // pick and merge pair of clusters, repeat N-1 times
    for (int i = 0; i < N - 1; i++)
    {
        last--; // decrement counter to ignore last element
        nextID++;
        int newNumBlocks =
            (int)ceil((FLOAT_TYPE)last / (FLOAT_TYPE)THREADSPERBLOCK);
        sycl::range<3> newGridN(newNumBlocks, 1, 1);
        // require shared memory for distances (tpb FLOAT_TYPES), clustID (tpb ints), and element A (tpb FLOAT_TYPES) 
        int sMem = sizeof(FLOAT_TYPE) * 2 * THREADSPERBLOCK + sizeof(int) * THREADSPERBLOCK;

        dpct::get_default_queue()
            .memcpy(clustID_d, clustID, sizeof(int) * (last + 1))
            .wait();

        // step1: get clustID for minimum distance
        // posB = getMinDistIndex(last + 1, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
        posB = getMinDistIndex(last + 1, closestDist_d, clustID_d, REDUCED_DISTS, REDUCED_INDICES);
        // get cluster ID of nearest neighbor
        posA = closestClust[posB];
        
        // update sequence of merged clusters
        seq[2 * i] = clustID[posA]; seq[2 * i + 1] = clustID[posB];
        
        // step2: merge elements and insert at A, update distances to A as necessary
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto dpct_global_range = gridD * block;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(dpct_global_range.get(2),
                                   dpct_global_range.get(1),
                                   dpct_global_range.get(0)),
                    sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    mergeElementsInsertAtA_CUDA(N, D, posA, posB, x_d,
                                                item_ct1);
                });
        });

        clustID[posA] = nextID;
        
        if (posA != 0) // no distances for first array position
        {
            // compute distances from A to preceding clusters and from following clusters to A
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(sMem), cgh);

                auto dpct_global_range = newGridN * block;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(block.get(2), block.get(1),
                                                     block.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        computeAllDistancesToA_CUDA(
                            N, last, D, posA, x_d, closestDist_d,
                            closestClust_d, REDUCED_DISTS, REDUCED_INDICES,
                            item_ct1, dpct_local_acc_ct1.get_pointer());
                    });
            });

            // update nearest neighbor heuristic for A
            updateDistanceAndIndexForCluster(posA, sMemReduce, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
        }
        
        // step3: replace cluster at B by last cluster
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
            auto dpct_global_range = gridD * block;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(dpct_global_range.get(2),
                                   dpct_global_range.get(1),
                                   dpct_global_range.get(0)),
                    sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    moveCluster_CUDA(N, D, posB, last, x_d, closestDist_d,
                                     closestClust_d, item_ct1);
                });
        });

        // move cluster ID
        clustID[posB] = clustID[last];
        
        // check if index of element N still relevant
        if (closestClust[last] < posB) closestClust[posB] = closestClust[last];
        else closestClust[posB] = -1;  // this means that distance will be updated below
        if (last > posB)
        {
            sycl::range<3> gridLargerThanB(
                (int)ceil((FLOAT_TYPE)(last - posB) /
                          (FLOAT_TYPE)THREADSPERBLOCK),
                1, 1); // require (last - posB) threads
            int sMem2 = sizeof(FLOAT_TYPE) * THREADSPERBLOCK;
            
            // check if new cluster at B changes stored values for neighbor heuristic for following clusters
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(sMem2), cgh);

                auto dpct_global_range = gridLargerThanB * block;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(block.get(2), block.get(1),
                                                     block.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        computeDistancesToBForPLargerThanB_CUDA(
                            N, D, posB, last - posB, x_d, closestDist_d,
                            closestClust_d, item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
            });
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
                int numBlocksJ =
                    (int)ceil((FLOAT_TYPE)j / (FLOAT_TYPE)THREADSPERBLOCK);
                sycl::range<3> gridJ(numBlocksJ, 1, 1);

                // update neighbor heuristic for cluster at j by checking all preceding clusters
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                                   sycl::access::target::local>
                        dpct_local_acc_ct1(sycl::range<1>(sMem), cgh);

                    auto dpct_global_range = gridJ * block;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(dpct_global_range.get(2),
                                           dpct_global_range.get(1),
                                           dpct_global_range.get(0)),
                            sycl::range<3>(block.get(2), block.get(1),
                                           block.get(0))),
                        [=](sycl::nd_item<3> item_ct1) {
                            recomputeMinDistanceForElementAt_j_CUDA(
                                N, D, j, x_d, closestDist_d, closestClust_d,
                                REDUCED_DISTS, REDUCED_INDICES, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                });

                // update nearest neighbor heuristic for j
                updateDistanceAndIndexForCluster(j, sMemReduce, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
            }
        }
        dpct::get_default_queue()
            .memcpy(closestClust, closestClust_d, sizeof(int) * last)
            .wait();
    }
    
    // free memory
    sycl::free(x_d, dpct::get_default_context());
    sycl::free(clustID_d, dpct::get_default_context());
    sycl::free(closestClust_d, dpct::get_default_context());
    sycl::free(closestDist_d, dpct::get_default_context());
    sycl::free(REDUCED_DISTS, dpct::get_default_context());
    sycl::free(REDUCED_INDICES, dpct::get_default_context());
    free(clustID);
    free(closestClust);
    
    return seq;
}

