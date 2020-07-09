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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "som_gpu.h"
#include <cmath>

using namespace std;


template <unsigned int BLOCKSIZE, class T, class U>
static void reduceMinTwo(int tid, T *s_A, U *s_B, sycl::nd_item<3> item_ct1)
{
    if (BLOCKSIZE >= 1024)
    {
        if (tid < 512)
        {
            // first line assures same sequence as sequential code; removing this feature can improve efficiency
            if (s_A[tid + 512] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 512]);
            if (s_A[tid + 512] < s_A[tid]) { s_A[tid] = s_A[tid + 512]; s_B[tid] = s_B[tid + 512]; }
        }
        item_ct1.barrier();
    }

    if (BLOCKSIZE >= 512) {
        if (tid < 256)
        {
            if (s_A[tid + 256] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 256]);
            if (s_A[tid + 256] < s_A[tid]) { s_A[tid] = s_A[tid + 256]; s_B[tid] = s_B[tid + 256]; }
        }
        item_ct1.barrier();
    }

    if (BLOCKSIZE >= 256) {
        if (tid < 128)
        {
            if (s_A[tid + 128] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 128]);
            if (s_A[tid + 128] < s_A[tid]) { s_A[tid] = s_A[tid + 128]; s_B[tid] = s_B[tid + 128]; }
        }
        item_ct1.barrier();
    }

    if (BLOCKSIZE >= 128) {
        if (tid < 64)
        {
            if (s_A[tid + 64] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 64]);
            if (s_A[tid + 64] < s_A[tid]) { s_A[tid] = s_A[tid + 64]; s_B[tid] = s_B[tid + 64]; }
        }
        item_ct1.barrier();
    }


    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { if (s_A[tid + 32] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 32]); if (s_A[tid + 32] < s_A[tid]) { s_A[tid] = s_A[tid + 32]; s_B[tid] = s_B[tid + 32]; } }
        if (BLOCKSIZE >= 32) { if (s_A[tid + 16] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 16]); if (s_A[tid + 16] < s_A[tid]) { s_A[tid] = s_A[tid + 16]; s_B[tid] = s_B[tid + 16]; } }
        if (BLOCKSIZE >= 16) { if (s_A[tid +  8] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 8]); if (s_A[tid +  8] < s_A[tid]) { s_A[tid] = s_A[tid +  8]; s_B[tid] = s_B[tid +  8]; } }
        if (BLOCKSIZE >=  8) { if (s_A[tid +  4] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 4]); if (s_A[tid +  4] < s_A[tid]) { s_A[tid] = s_A[tid +  4]; s_B[tid] = s_B[tid +  4]; } }
        if (BLOCKSIZE >=  4) { if (s_A[tid +  2] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 2]); if (s_A[tid +  2] < s_A[tid]) { s_A[tid] = s_A[tid +  2]; s_B[tid] = s_B[tid +  2]; } }
        if (BLOCKSIZE >=  2) { if (s_A[tid +  1] == s_A[tid])
                s_B[tid] = sycl::min(s_B[tid], s_B[tid + 1]); if (s_A[tid +  1] < s_A[tid]) { s_A[tid] = s_A[tid +  1]; s_B[tid] = s_B[tid +  1]; } }
    }
}



static void findBMU_CUDA(int N, int K, int D, int v, FLOAT_TYPE *X, FLOAT_TYPE *WV, int *BMU, FLOAT_TYPE *DISTS,
                         sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto array = (float *)dpct_local; // declares variable in shared memory
    FLOAT_TYPE *s_data   = (FLOAT_TYPE*) array;                  // dynamically allocate array to hold position data for tpb FLOAT_TYPEs
    FLOAT_TYPE *s_dist = (FLOAT_TYPE *)&s_data[item_ct1.get_local_range().get(
        2)]; // dynamically allocate array to hold intermediate distance results
    FLOAT_TYPE *s_inputV = (FLOAT_TYPE *)&s_dist[item_ct1.get_local_range().get(
        2)]; // dynamically allocate variable to hold components of input vector

    int t = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);   // global thread ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID

    s_dist[tid] = FLT_MAX;
    if (t < K) s_dist[tid] = 0.0;
    
    // for each dimension
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb components of input vector to shared memory (non-coalesced)
        if (offsetD + tid < D) s_inputV[tid] = X[(offsetD + tid) * N + v];
        item_ct1.barrier();
        // compute distances between up to tpb weight vectors and input vector components
        for (unsigned int d = offsetD;
             d <
             sycl::min((unsigned int)(offsetD + item_ct1.get_local_range(2)),
                       (unsigned int)D);
             d++)
        {
            if (t < K) s_dist[tid] += distanceComponentGPU(WV + d * K + t, s_inputV + d - offsetD);
        }
        offsetD += item_ct1.get_local_range().get(2);
        item_ct1.barrier();
    }
    s_dist[tid] = distanceFinalizeGPU(1, s_dist + tid); 
    
    // now s_dist contains blockDim.x distances; reduce to find best matching unit in block     
    
    // reuse s_data
    s_data[tid] = tid;
    item_ct1.barrier();

    // reduce sdata and iresult, to get minimum distance and bmu index
    reduceMinTwo<THREADSPERBLOCK>(tid, s_dist, s_data, item_ct1);

    // return values
    if (tid == 0)
    {
        BMU[item_ct1.get_group(2)] = (int)s_data[tid];
        DISTS[item_ct1.get_group(2)] = s_dist[tid];
    }      
}



static void copyBMU_CUDA(int K, int D, int bmu, FLOAT_TYPE *WV, FLOAT_TYPE *BMU,
                         sycl::nd_item<3> item_ct1)
{
    int tid = item_ct1.get_local_id(2);
    if (tid < D)
    {
        // not coalesced
        BMU[tid] = WV[tid * K + bmu];
    }
}


static void updateNeighborhood_CUDA(int N, int K, int D, int v, FLOAT_TYPE d0, FLOAT_TYPE scale, FLOAT_TYPE *BMU, FLOAT_TYPE *X, FLOAT_TYPE *WV,
                                    sycl::nd_item<3> item_ct1,
                                    uint8_t *dpct_local)
{
    auto array = (float *)dpct_local; // declares variable in shared memory
    FLOAT_TYPE *s_bmuDist    = (FLOAT_TYPE*) array; // dynamically allocate array to hold intermediate distance results
    FLOAT_TYPE *s_vectorComp =
        (FLOAT_TYPE *)&s_bmuDist[item_ct1.get_local_range().get(
            2)]; // dynamically allocate variable to hold components of input
                 // vector

    int t = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);   // global thread ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID

    s_bmuDist[tid] = 0.0;
    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        // copy up to tpb bmu vector components to shared memory (coalesced)
        if (offsetD + tid < D) s_vectorComp[tid] = BMU[offsetD + tid];
        item_ct1.barrier();
        // compute distances for up to tpb dimensions
        for (unsigned int d = offsetD;
             d <
             sycl::min((unsigned int)(offsetD + item_ct1.get_local_range(2)),
                       (unsigned int)D);
             d++)
        {
            if (t < K) s_bmuDist[tid] += distanceComponentGPU(WV + d * K + t, s_vectorComp + d - offsetD);
        }
        offsetD += item_ct1.get_local_range().get(2);
        item_ct1.barrier();
    }
    if (t < K) s_bmuDist[tid] = distanceFinalizeGPU(1, s_bmuDist + tid);
    item_ct1.barrier();
    // now update weight vector position towards inputV using d0, bmuDist, and learning restraint
    offsetD = 0;
    while (offsetD < D)
    {
        // read up to tpb components from input vector (non-coalesced)
        if (offsetD + tid < D) s_vectorComp[tid] = X[(offsetD + tid) * N + v];
        item_ct1.barrier();
        // modify up to tpb components of up to tpb weight vectors (coalesced)
        for (unsigned int d = offsetD;
             d <
             sycl::min((unsigned int)(offsetD + item_ct1.get_local_range(2)),
                       (unsigned int)D);
             d++)
        {
            if (t < K)
                WV[d * K + t] +=
                    scale *
                    sycl::pow((FLOAT_TYPE)0.25, (FLOAT_TYPE)(s_bmuDist[tid] / d0)) *
                    (s_vectorComp[d - offsetD] - WV[d * K + t]);
        }
        offsetD += item_ct1.get_local_range().get(2);
        item_ct1.barrier();
    }
}



void somGPU(int N, int K, int D, int numIter, FLOAT_TYPE *x, FLOAT_TYPE **pwv, DataIO *data)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    FLOAT_TYPE *wv = *pwv;
    
    // determine CUDA parameters
    sycl::range<3> block(THREADSPERBLOCK, 1, 1);
    sycl::range<3> gridK2(
        (int)ceil((FLOAT_TYPE)K / (FLOAT_TYPE)THREADSPERBLOCK), 1, 1);
    int numBlocks = (int)ceil((FLOAT_TYPE)K / (FLOAT_TYPE)THREADSPERBLOCK);
    sycl::range<3> gridK(numBlocks, 1, 1);
    sycl::range<3> gridD((int)ceil((FLOAT_TYPE)D / (FLOAT_TYPE)THREADSPERBLOCK),
                         1, 1);
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
            try {
                q_ct1.submit([&](sycl::handler& cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                        sycl::access::target::local>
                        dpct_local_acc_ct1(sycl::range<1>(sMemBMU), cgh);

                    auto dpct_global_range = gridK * block;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                         dpct_global_range.get(1),
                                                         dpct_global_range.get(0)),
                                          sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                        [=](sycl::nd_item<3> item_ct1) {
                            findBMU_CUDA(N, K, D, v, (FLOAT_TYPE*)x_d,
                                (FLOAT_TYPE*)wv_d, (int*)bmuIDs_d,
                                (FLOAT_TYPE*)dists_d, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                    }).wait_and_throw();
            }
            catch (cl::sycl::exception const& e) {
                std::cout << "Caught SYCL exception (findBMU kernel):\n"
                    << e.what() << std::endl;
            }
            
            // finish reduction on CPU
            q_ct1.memcpy(bmuIDs, (int *)bmuIDs_d, sizeof(int) * numBlocks)
                .wait();
            q_ct1
                .memcpy(dists, (FLOAT_TYPE *)dists_d,
                        sizeof(FLOAT_TYPE) * numBlocks)
                .wait();
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
            try {
                q_ct1.submit([&](sycl::handler& cgh) {
                    auto dpct_global_range = gridD * block;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                            dpct_global_range.get(1),
                            dpct_global_range.get(0)),
                            sycl::range<3>(block.get(2), block.get(1),
                                block.get(0))),
                        [=](sycl::nd_item<3> item_ct1) {
                            copyBMU_CUDA(K, D, bmuID, (FLOAT_TYPE*)wv_d,
                                (FLOAT_TYPE*)bmu_d, item_ct1);
                        });
                    });
            }
            catch (cl::sycl::exception const& e) {
                std::cout << "Caught SYCL exception (copyBMU kernel):\n"
                    << e.what() << std::endl;
            }
            
            // moves wv towards input vector
            try {
                q_ct1.submit([&](sycl::handler& cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                        sycl::access::target::local>
                        dpct_local_acc_ct1(sycl::range<1>(sMemNei), cgh);

                    auto dpct_global_range = gridK * block;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                            dpct_global_range.get(1),
                            dpct_global_range.get(0)),
                            sycl::range<3>(block.get(2), block.get(1),
                                block.get(0))),
                        [=](sycl::nd_item<3> item_ct1) {
                            updateNeighborhood_CUDA(
                                N, K, D, v, minDist, learningRestraint,
                                (FLOAT_TYPE*)bmu_d, (FLOAT_TYPE*)x_d,
                                (FLOAT_TYPE*)wv_d, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                    });
            }
            catch (cl::sycl::exception const& e) {
                std::cout << "Caught SYCL exception (updateNeighborhood kernel):\n"
                    << e.what() << std::endl;
            }
        }
    }
    
    // copy weight vector values back to the host
    q_ct1.memcpy(wv, wv_d, sizeof(FLOAT_TYPE) * K * D).wait();

    // free memory
    try {
        sycl::free(bmu_d, q_ct1);
        sycl::free(wv_d, q_ct1);
        sycl::free(x_d, q_ct1);
    }
    catch (cl::sycl::exception const& e) {
        std::cout << "Caught SYCL exception (free memory):\n"
            << e.what() << std::endl;
    }

    *pwv = wv;
}

