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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kpsmeans_gpu.h"
#include <cmath>

using namespace std;

template <unsigned int BLOCKSIZE, class T>
static void reduceOne(int tid, T* s_A,
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

template <unsigned int BLOCKSIZE, class T, class U>
static void reduceTwo(int tid, T* s_A, U* s_B, sycl::nd_item<3> item_ct1)
{
    for (unsigned idx{ 1024U }; idx > 1; idx /= 2)
    {
        const auto halfIdx = idx / 2u;
        if (BLOCKSIZE >= idx)
        {
            if (tid < halfIdx)
            {
                s_A[tid] += s_A[tid + halfIdx];
                s_B[tid] += s_B[tid + halfIdx];
            }
            item_ct1.barrier();
        }
    }
}


static void assignToClusters_KPSMCUDA(int N, int K, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local)
{
    auto* array = (float*)dpct_local;
    auto* s_center = (FLOAT_TYPE*)array;

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



static void calcScoreSorted_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, FLOAT_TYPE* SCORE, int* SEGOFFSET,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local)
{
    auto* array = (float*)dpct_local;
    auto* s_scores = (FLOAT_TYPE*)array;
    auto* s_center = (FLOAT_TYPE*)&s_scores[item_ct1.get_local_range(2)];
    auto* s_segment = (int*)&s_center[item_ct1.get_local_range(2)];

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

            const auto minValue = sycl::min((unsigned int)(offsetD + item_ct1.get_local_range(2)), (unsigned int)D);
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

static void calcScore_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN, FLOAT_TYPE* SCORE,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local)
{
    auto array = (float*)dpct_local; // shared memory
    FLOAT_TYPE* s_scores = (FLOAT_TYPE*)array;                    // tpb partial scores
    FLOAT_TYPE* s_center =
        (FLOAT_TYPE*)&s_scores[item_ct1.get_local_range().get(
            2)]; // up to tpb centroid components

    int k = item_ct1.get_group(2);      // cluster ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID

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
            item_ct1.barrier();
            // thread divergence likely
            if (ASSIGN[offsetN] == k)
            {
                // for each of the following tpb (or D - offsetD) dimensions
                for (unsigned int d = offsetD;
                    d < sycl::min((unsigned int)(offsetD +
                        item_ct1.get_local_range(2)),
                        (unsigned int)D);
                    d++)
                {
                    // broadcast centroid component and compute contribution to distance to data point
                    dist += distanceComponentGPU(s_center + (d - offsetD), X + (d * N + offsetN));
                }
            }
            offsetD += item_ct1.get_local_range().get(2);
            item_ct1.barrier();
        }
        // update partial score
        s_scores[tid] += distanceFinalizeGPU(1, &dist);
        offsetN += item_ct1.get_local_range().get(2);
    }
    item_ct1.barrier();
    // compute score for block by reducing over threads
    reduceOne<THREADSPERBLOCK>(tid, s_scores, item_ct1);
    if (tid == 0) SCORE[k] = s_scores[tid];
}



static void calcCentroidsSorted_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* SEGOFFSET,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local)
{
    auto array = (float*)dpct_local;                           // shared memory
    int* s_numElements = (int*)array;                      // tpb partial sums of elements in cluster
    FLOAT_TYPE* s_centerParts =
        (FLOAT_TYPE*)&s_numElements[item_ct1.get_local_range().get(
            2)]; // tpb partial centroid components
    int* s_segment = (int*)&s_centerParts[item_ct1.get_local_range().get(2)];

    int k = item_ct1.get_group(2);      // centroid ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID

    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];           // transfer start and end offsets to shared memory
    item_ct1.barrier();
    int writeOffset = s_segment[0] + tid;                       // broadcast segment start offset to registers of all threads and add unique tid
    int endOffset = s_segment[1];                             // broadcast segment end offset to registers of all threads

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
            offset += item_ct1.get_local_range().get(2);
        }
        item_ct1.barrier();

        // take sum over all tpb array elements
        // reduce number of cluster elements only once
        if (d == 0)
        {
            // reduce number of elements and centroid parts
            reduceTwo<THREADSPERBLOCK>(tid, s_centerParts, s_numElements,
                item_ct1);
            if (tid == 0) clusterSize = (FLOAT_TYPE)s_numElements[tid];
        }
        else
        {
            // reduce centroid parts
            reduceOne<THREADSPERBLOCK>(tid, s_centerParts, item_ct1);
        }
        // write result to global mem (non-coalesced)
        // could replace this by coalesced writes followed by matrix transposition
        if (tid == 0) if (clusterSize > 0) CTR[k * D + d] = s_centerParts[tid] / clusterSize;
    }
}



static void calcCentroids_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local)
{
    auto array = (float*)dpct_local;                           // shared memory
    int* s_numElements = (int*)array;                      // tpb partial sums of elements in cluster
    FLOAT_TYPE* s_centerParts =
        (FLOAT_TYPE*)&s_numElements[item_ct1.get_local_range().get(
            2)]; // tpb partial centroid components

    int k = item_ct1.get_group(2);      // cluster ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID

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
            offset += item_ct1.get_local_range().get(2);
        }
        item_ct1.barrier();

        // take sum over all tpb array elements
        // reduce number of cluster elements only once
        if (d == 0)
        {
            // reduce number of elements and centroid parts
            reduceTwo<THREADSPERBLOCK>(tid, s_centerParts, s_numElements,
                item_ct1);
            if (tid == 0) clusterSize = (FLOAT_TYPE)s_numElements[tid];
            // note: if clusterSize == 0 we can return here
        }
        else
        {
            // reduce centroid parts
            reduceOne<THREADSPERBLOCK>(tid, s_centerParts, item_ct1);
        }
        // write result to global mem (non-coalesced)
        // could replace this by coalesced writes followed by matrix transposition
        if (tid == 0) if (clusterSize > 0) CTR[k * D + d] = s_centerParts[tid] / clusterSize;
    }
}



template <unsigned int BLOCKSIZE>
static int parallelPrefixSum(int tid, int* DATA, sycl::nd_item<3> item_ct1)
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



static void sort_getSegmentSize_CUDA(int N, int* ASSIGN, int* SEGSIZE,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local)
{
    // requires tpb*sizeof(int) bytes of shared memory
    auto array = (float*)dpct_local;                  // shared memory
    int* s_num = (int*)array;                  // tpb partial segment sizes

    unsigned int k = item_ct1.get_group(2);      // cluster ID
    unsigned int tid = item_ct1.get_local_id(2); // thread ID in block
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
    item_ct1.barrier();
    // now reduce sum over threads in block
    reduceOne<THREADSPERBLOCK>(tid, s_num, item_ct1);
    // output segment size for cluster
    if (tid == 0) SEGSIZE[k] = s_num[tid];
}


static void sort_moveData_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGOFFSET,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local)
{
    // requires (5*tpb+2)*sizeof(int) Bytes of shared memory
    auto array = (float*)dpct_local;                           // shared memory
    int* s_gather = (int*)array;                           // up to 3 * tpb assignments
    int* s_indices = (int*)&s_gather[3 * THREADSPERBLOCK];  // 2 * tpb intermediate results for parallel all-prefix sum
    int* s_segment = (int*)&s_indices[2 * THREADSPERBLOCK]; // 2 integer numbers to hold start and end offsets of segment for broadcasting

    bool scan1 = false;
    bool scan2 = false;

    int k = item_ct1.get_group(2);      // cluster ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID
    s_gather[tid] = 0;
    s_gather[tid + THREADSPERBLOCK] = 0;
    s_gather[tid + 2 * THREADSPERBLOCK] = 0;
    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];           // transfer start and end offsets to shared memory
    item_ct1.barrier();
    int bufferOffset = s_segment[0] + tid;                      // broadcast segment start offset to registers of all threads and add unique tid
    int bufferEndOffset = s_segment[1];                       // broadcast segment end offset to registers of all threads
    int dataOffset = tid;                                       // offset in data set
    int windowSize = 2 * THREADSPERBLOCK;                       // number of data points processed per iterations
    int numFound = 0;                                         // number of data points collected

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
        scan1 = ((dataOffset < N) && (ASSIGN[dataOffset] == k));
        // not yet end of data array and data point at tid + tpb assigned to cluster?
        scan2 = ((dataOffset + THREADSPERBLOCK < N) && (ASSIGN[dataOffset + THREADSPERBLOCK] == k));
        // set shared memory array to 1 for data points in segment assigned to cluster k, and to 0 otherwise
        s_indices[tid] = 0;
        s_indices[tid + THREADSPERBLOCK] = 0;
        if (scan1) s_indices[tid] = 1;
        if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;

        // returns unique indices for data points assigned to cluster and total number of data points found in segment
        int nonZero =
            parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices, item_ct1);

        // check which (if any) data points belong to cluster and store indices of those found
        if (scan1) s_gather[numFound + s_indices[tid]] = dataOffset;
        if (scan2) s_gather[numFound + s_indices[tid + THREADSPERBLOCK]] = dataOffset + THREADSPERBLOCK;
        // update number of data points found but not yet transferred to buffer
        numFound += nonZero;

        item_ct1.barrier();
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
            s_gather[tid] = s_gather[tid + THREADSPERBLOCK];
            s_gather[tid + THREADSPERBLOCK] = s_gather[tid + 2 * THREADSPERBLOCK];
            s_gather[tid + 2 * THREADSPERBLOCK] = 0;
            // move to next buffer segment
            bufferOffset += THREADSPERBLOCK;
        }
        // move to next data segment
        dataOffset += windowSize;
    }
    item_ct1.barrier();

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
void serialPrefixSum_KPSMCUDA(int N, int* INPUT, int* OUTPUT)
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
    // transfer data to host
    int* intermediates = (int*)malloc(sizeof(int) * N);
    q_ct1.memcpy(intermediates, INPUT, sizeof(int) * N).wait();
    // value at location 0 is okay
    for (unsigned int i = 1; i < N; i++)
    {
        intermediates[i] += intermediates[i - 1];
    }
    // transfer results to device
    q_ct1.memcpy(OUTPUT, intermediates, sizeof(int) * N).wait();
    free(intermediates);
}



static void upd_getReassigns_CUDA(int N, int* ASSIGN, int* SEGOFFSET, int* SEGSIZE, int* UPDSEGSIZE,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local)
{
    // requires (tpb+2)*sizeof(int) bytes of shared memory
    auto array = (float*)dpct_local;                       // shared memory
    int* s_num = (int*)array;                      // tpb partial sums of wrongly assigned data points
    int* s_segment = (int*)&s_num[THREADSPERBLOCK];    // intermediate storage for broadcasting segment boundaries

    unsigned int k = item_ct1.get_group(2);      // cluster ID
    unsigned int tid = item_ct1.get_local_id(2); // in-block thread ID
    unsigned int num = 0;

    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];       // transfer start and end offsets to shared memory
    item_ct1.barrier();
    int endOffset = s_segment[1];                         // broadcast segment end offset to registers of all threads

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
    item_ct1.barrier();
    // now reduce sum over threads in block
    reduceOne<THREADSPERBLOCK>(tid, s_num, item_ct1);
    if (tid == 0)
    {
        // output segment size for block
        UPDSEGSIZE[k] = s_num[tid];
        // update data segment size for block
        SEGSIZE[k] -= s_num[tid];
    }
}



static void upd_moveAssigns_CUDA(int N, int* ASSIGN, int* UPDASSIGN, int* SEGOFFSET, int* UPDSEGOFFSET,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local)
{
    // requires (5*tpb+4)*sizeof(int) Bytes of shared memory
    auto array = (float*)dpct_local;                           // shared memory
    int* s_gather = (int*)array;                           // up to 3 * tpb assignments
    int* s_indices = (int*)&s_gather[3 * THREADSPERBLOCK];  // 2 * tpb intermediate results for parallel all-prefix sum
    int* s_segment = (int*)&s_indices[2 * THREADSPERBLOCK]; // 4 integer numbers to hold start and end offsets of segments for broadcasting

    bool scan1 = false;
    bool scan2 = false;

    int k = item_ct1.get_group(2);      // cluster ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID

    s_gather[tid] = 0;
    s_gather[tid + THREADSPERBLOCK] = 0;
    s_gather[tid + 2 * THREADSPERBLOCK] = 0;
    if (tid < 2)
    {
        s_segment[tid] = UPDSEGOFFSET[k + tid];                 // transfer start and end offset of buffer segment to shared memory
        s_segment[tid + 2] = SEGOFFSET[k + tid];                // transfer start and end offset of assignment segment to shared memory
    }
    item_ct1.barrier();
    int bufferOffset = s_segment[0] + tid;                      // broadcast segment start offset to registers of all threads and add unique tid
    int bufferEndOffset = s_segment[1];                        // broadcast segment end offset to registers of all threads
    int dataOffset = s_segment[2] + tid;                        // broadcast assignment start offset
    int dataEndOffset = s_segment[3];                           // broadcast assignment end offset
    int windowSize = 2 * THREADSPERBLOCK;                       // number of data points processed per iteration
    int numFound = 0;                                         // number of data points collected

    // completes when all misassigned data points are found
    while ((dataOffset - tid) < dataEndOffset && (bufferOffset - tid + numFound) < bufferEndOffset)
    {
        // collect tpb indices using string compaction, then move data
        // convert windowSize assignments to binary: 1 = not assigned, or 0 = assigned to current cluster

        // data point at tid not assigned to cluster?
        scan1 = ((dataOffset < dataEndOffset) && (ASSIGN[dataOffset] != k));
        // not yet end of array and data point at tid + tpb not assigned to cluster?
        scan2 = ((dataOffset + THREADSPERBLOCK < dataEndOffset) && (ASSIGN[dataOffset + THREADSPERBLOCK] != k));
        // set shared memory to 1 for data points not assigned to cluster k, and to 0 otherwise
        s_indices[tid] = 0;
        s_indices[tid + THREADSPERBLOCK] = 0;
        if (scan1) s_indices[tid] = 1;
        if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;

        // returns unique indices for data points not assigned to cluster and total number of data points found in segment
        int nonZero =
            parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices, item_ct1);

        // check which (if any) data points do not belong to cluster and store indices of those found
        if (scan1) s_gather[numFound + s_indices[tid]] = dataOffset;
        if (scan2) s_gather[numFound + s_indices[tid + THREADSPERBLOCK]] = dataOffset + THREADSPERBLOCK;
        // update number of data points found but not yet transferred to buffer
        numFound += nonZero;

        item_ct1.barrier();
        // while we have enough indices to do efficient coalesced memory writes
        while (numFound >= THREADSPERBLOCK)
        {
            // transfer assignment to buffer
            UPDASSIGN[bufferOffset] = ASSIGN[s_gather[tid]];
            // update number of assignments found but not yet transferred
            numFound -= THREADSPERBLOCK;
            // move down collected indices of assignments and overwrite those already used
            s_gather[tid] = s_gather[tid + THREADSPERBLOCK];
            s_gather[tid + THREADSPERBLOCK] = s_gather[tid + 2 * THREADSPERBLOCK];
            s_gather[tid + 2 * THREADSPERBLOCK] = 0;
            // move to next buffer segment
            bufferOffset += THREADSPERBLOCK;
        }
        // move to next data segment
        dataOffset += windowSize;
    }
    item_ct1.barrier();
    // now write the remaining assignments to buffer (writing coalesced, but not necessarily complete set of threads involved)
    if (bufferOffset < bufferEndOffset)
    {
        // transfer assignment
        UPDASSIGN[bufferOffset] = ASSIGN[s_gather[tid]];
    }
}



static void upd_segmentSize_CUDA(int* UPDASSIGN, int* BUFFERSIZE, int* SEGSIZE,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local)
{
    // requires (tpb+1)*sizeof(int) bytes of shared memory
    auto array = (float*)dpct_local;                      // shared memory
    int* s_num = (int*)array;                     // tpb partial sums of assignments to cluster k in buffer
    int* s_segment = (int*)&s_num[THREADSPERBLOCK];

    unsigned int k = item_ct1.get_group(2);      // cluster ID
    unsigned int tid = item_ct1.get_local_id(2); // thread ID in block
    unsigned int num = 0;

    if (tid == 0) s_segment[tid] = BUFFERSIZE[0];          // transfer end offset to shared memory
    item_ct1.barrier();
    int endOffset = s_segment[0];                        // broadcast segment end offset to registers of all threads

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
    item_ct1.barrier();
    // now reduce sum over threads in block
    reduceOne<THREADSPERBLOCK>(tid, s_num, item_ct1);
    // output new data segment size for cluster
    if (tid == 0) SEGSIZE[k] += s_num[tid];
}



static void upd_findWrongAssigns_CUDA(int N, int* ASSIGN, int* SEGOFFSET, int* UPDSEGSIZE,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local)
{
    // requires (tpb+2)*sizeof(int) bytes of shared memory
    auto array = (float*)dpct_local;                        // shared memory
    int* s_num = (int*)array;                       // tpb partial segment sizes
    int* s_segment = (int*)&s_num[THREADSPERBLOCK];     // intermediate storage for broadcasting segment boundaries

    unsigned int k = item_ct1.get_group(2);      // cluster ID
    unsigned int tid = item_ct1.get_local_id(2); // in-block thread ID
    unsigned int num = 0;

    if (tid < 2) s_segment[tid] = SEGOFFSET[k + tid];        // transfer start and end offsets to shared memory
    item_ct1.barrier();
    int endOffset = s_segment[1];                          // broadcast segment end offset to registers of all threads

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
    item_ct1.barrier();
    // now reduce sum over threads in block
    reduceOne<THREADSPERBLOCK>(tid, s_num, item_ct1);
    // output segment size for cluster
    if (tid == 0) UPDSEGSIZE[k] = s_num[tid];
}



static void upd_moveDataToBuffer_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGOFFSET, int* UPDSEGOFFSET,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local)
{
    // requires (5*tpb+4)*sizeof(int) Bytes of shared memory
    auto array = (float*)dpct_local;                           // shared memory
    int* s_gather = (int*)array;                           // up to 3 * tpb assignments
    int* s_indices = (int*)&s_gather[3 * THREADSPERBLOCK];  // 2 * tpb intermediate results for parallel all-prefix sum
    int* s_segment = (int*)&s_indices[2 * THREADSPERBLOCK]; // 4 integer numbers to hold start and end offsets of buffer and data segments for broadcasting

    bool scan1 = false;
    bool scan2 = false;

    int k = item_ct1.get_group(2);      // cluster ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID
    s_gather[tid] = 0;
    s_gather[tid + THREADSPERBLOCK] = 0;
    s_gather[tid + 2 * THREADSPERBLOCK] = 0;
    if (tid < 2)
    {
        s_segment[tid] = UPDSEGOFFSET[k + tid];                 // transfer start and end offsets of buffer segment to shared memory
        s_segment[tid + 2] = SEGOFFSET[k + tid];                // transfer start and end offsets of data segment to shared memory
    }
    item_ct1.barrier();
    int bufferOffset = s_segment[0] + tid;                      // broadcast buffer segment start offset to registers of all threads and add unique tid
    int bufferEndOffset = s_segment[1];                       // broadcast buffer segment end offset to registers of all threads
    int dataOffset = s_segment[2] + tid;                        // offset in data set
    int dataEndOffset = s_segment[3];
    int windowSize = 2 * THREADSPERBLOCK;                       // number of data points processed per iterations
    int numFound = 0;                                         // number of data points collected

    // completes when all misassigned data points have been found
    while ((dataOffset - tid) < dataEndOffset && (bufferOffset - tid + numFound) < bufferEndOffset)
    {
        // collect tpb indices using string compaction, then move data
        // move windowSize assignments to binary: 1 = not assigned, or 0 = assigned to current cluster

        // data point at tid not assigned to cluster?
        scan1 = ((dataOffset < dataEndOffset) && (ASSIGN[dataOffset] != k));
        // not yet end of array and data point at tid + tpb not assigned to cluster?
        scan2 = ((dataOffset + THREADSPERBLOCK < dataEndOffset) && (ASSIGN[dataOffset + THREADSPERBLOCK] != k));
        // set shared memory to 1 for data points not assigned to cluster k, and to 0 otherwise
        s_indices[tid] = 0;
        s_indices[tid + THREADSPERBLOCK] = 0;
        if (scan1) s_indices[tid] = 1;
        if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;

        // returns unique indices for data points not assigned to cluster and total number of data points found in segment
        int nonZero =
            parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices, item_ct1);

        // check which (if any) data points do not belong to cluster and store indices of those found
        if (scan1) s_gather[numFound + s_indices[tid]] = dataOffset;
        if (scan2) s_gather[numFound + s_indices[tid + THREADSPERBLOCK]] = dataOffset + THREADSPERBLOCK;
        // update number of data points found but not yet transferred to buffer
        numFound += nonZero;

        item_ct1.barrier();
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
            s_gather[tid] = s_gather[tid + THREADSPERBLOCK];
            s_gather[tid + THREADSPERBLOCK] = s_gather[tid + 2 * THREADSPERBLOCK];
            s_gather[tid + 2 * THREADSPERBLOCK] = 0;
            // move to next buffer segment
            bufferOffset += THREADSPERBLOCK;
        }
        // move to next data segment
        dataOffset += windowSize;
    }
    item_ct1.barrier();
    // now write the remaining data points to buffer (writing coaelesced, but not necessarily all threads involved)
    if (bufferOffset < bufferEndOffset)
    {
        // transfer data point
        for (int d = 0; d < D; d++) X2[bufferOffset + d * N] = X[s_gather[tid] + d * N];
        // transfer assignment
        UPDASSIGN[bufferOffset] = ASSIGN[s_gather[tid]];
    }
}



static void upd_collectDataFromBuffer_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGOFFSET, int* BUFFERSIZE,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local)
{
    // requires (8*tpb)*sizeof(int) Bytes of shared memory
    auto array = (float*)dpct_local; // shared memory
    int* s_gatherA = (int*)array;                           // up to 3 * tpb assignments
    int* s_gatherB = (int*)&s_gatherA[3 * THREADSPERBLOCK]; // up to 3 * tpb assignments
    int* s_indices = (int*)&s_gatherB[3 * THREADSPERBLOCK]; // 2 * tpb intermediate results for parallel all-prefix sum

    bool scan1 = false;
    bool scan2 = false;

    int k = item_ct1.get_group(2);      // cluster ID
    int tid = item_ct1.get_local_id(2); // in-block thread ID
    s_gatherA[tid] = 0;
    s_gatherA[tid + THREADSPERBLOCK] = 0;
    s_gatherA[tid + 2 * THREADSPERBLOCK] = 0;
    s_gatherB[tid] = 0;
    s_gatherB[tid + THREADSPERBLOCK] = 0;
    s_gatherB[tid + 2 * THREADSPERBLOCK] = 0;
    if (tid < 2) s_indices[tid] = SEGOFFSET[k + tid];             // transfer start and end offsets of data segment to shared memory
    if (tid < 1) s_indices[tid + 2] = BUFFERSIZE[0];              // transfer end offset of buffer segment to shared memory
    item_ct1.barrier();
    int bufferOffset = tid;                                       // start at first position of buffer
    int bufferEndOffset = s_indices[2];                           // broadcast buffer end offset to registers of all threads
    int dataOffset = s_indices[0] + tid;                          // broadcast data start offset to registers of all threads and add unique tid
    int dataEndOffset = s_indices[1];                             // broadcast data end offset to registers of all threads
    item_ct1.barrier();
    int numFoundData = 0;                                          // number of gaps in data array found
    int numFoundBuffer = 0;                                       // number of data points in buffer found

    // run through segment assigned to current cluster center and fill in the missing data points
    while ((dataOffset - tid) < dataEndOffset || (bufferOffset - tid) < bufferEndOffset)
    {
        // collect tpb indices from data array and tpb indices from buffer using string compaction, then move data
        // move windowSize assignments to shared memory coded as 0 (not assigned to current cluster) or 1 (assigned)
        while ((dataOffset - tid) < dataEndOffset && numFoundData < THREADSPERBLOCK)
        {
            // data point at tid not assigned to cluster?
            scan1 = ((dataOffset < dataEndOffset) && (ASSIGN[dataOffset] != k));
            // not yet end of array and data point at tid + tpb not assigned to cluster?
            scan2 = ((dataOffset + THREADSPERBLOCK < dataEndOffset) && (ASSIGN[dataOffset + THREADSPERBLOCK] != k));
            // set shared memory to 1 for data points not assigned to cluster k, and to 0 otherwise
            s_indices[tid] = 0;
            s_indices[tid + THREADSPERBLOCK] = 0;
            if (scan1) s_indices[tid] = 1;
            if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;

            // returns unique indices for data points not assigned to cluster and total number of such data points found in segment
            int nonZero =
                parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices, item_ct1);

            // check which (if any) data points do not belong to cluster and store indices of those found
            if (scan1) s_gatherA[numFoundData + s_indices[tid]] = dataOffset;
            if (scan2) s_gatherA[numFoundData + s_indices[tid + THREADSPERBLOCK]] = dataOffset + THREADSPERBLOCK;
            // update number of data points found but not yet transferred from buffer
            numFoundData += nonZero;
            // move to next data segment
            dataOffset += 2 * THREADSPERBLOCK;
        }
        // one of two options: we have collected enough indices to do memory transfer using all threads, or we have reached the end of the segment
        item_ct1.barrier();

        // now, do the same for the buffer data
        while ((bufferOffset - tid) < bufferEndOffset && numFoundBuffer < THREADSPERBLOCK)
        {
            // data point at tid assigned to cluster?
            scan1 = ((bufferOffset < bufferEndOffset) && (UPDASSIGN[bufferOffset] == k));
            // not yet end of array and data point at tid + tpb assigned to cluster?
            scan2 = ((bufferOffset + THREADSPERBLOCK < bufferEndOffset) && (UPDASSIGN[bufferOffset + THREADSPERBLOCK] == k));
            // set shared memory to 1 for data points assigned to cluster k, and to 0 otherwise
            s_indices[tid] = 0;
            s_indices[tid + THREADSPERBLOCK] = 0;
            if (scan1) s_indices[tid] = 1;
            if (scan2) s_indices[tid + THREADSPERBLOCK] = 1;

            // returns unique indices for data points assigned to cluster and total number of such data points found in segment
            int nonZero =
                parallelPrefixSum<THREADSPERBLOCK>(tid, s_indices, item_ct1);

            // check which (if any) data points belong to cluster and store indices of those found
            if (scan1) s_gatherB[numFoundBuffer + s_indices[tid]] = bufferOffset;
            if (scan2) s_gatherB[numFoundBuffer + s_indices[tid + THREADSPERBLOCK]] = bufferOffset + THREADSPERBLOCK;
            // update number of data points found but not yet transferred to data array
            numFoundBuffer += nonZero;
            // move to next buffer segment
            bufferOffset += 2 * THREADSPERBLOCK;
        }
        // one of two options: we have collected enough indices to do memory transfer using all threads, or we have reached the end of the buffer

        item_ct1.barrier();

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
            s_gatherA[tid] = s_gatherA[tid + THREADSPERBLOCK];
            s_gatherB[tid] = s_gatherB[tid + THREADSPERBLOCK];
            s_gatherA[tid + THREADSPERBLOCK] = s_gatherA[tid + 2 * THREADSPERBLOCK];
            s_gatherB[tid + THREADSPERBLOCK] = s_gatherB[tid + 2 * THREADSPERBLOCK];
            s_gatherA[tid + 2 * THREADSPERBLOCK] = 0;
            s_gatherB[tid + 2 * THREADSPERBLOCK] = 0;
        }
    }
    item_ct1.barrier();
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


bool updateData(int N, int K, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGSIZE, int* UPDSEGSIZE, int* SEGOFFSET, int* UPDSEGOFFSET)
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
    sycl::range<3> block(THREADSPERBLOCK, 1,
        1);       // set number of threads per block
    sycl::range<3> gridK(K, 1, 1); // K blocks of threads in grid

    // step 1: Find number of data points wrongly assigned and update number of data points, updates SEGSIZE and UPDSEGSIZE
    int sMem = sizeof(int) * (THREADSPERBLOCK + 2);
    q_ct1.submit([&](sycl::handler& cgh) {
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
                upd_getReassigns_CUDA(N, ASSIGN, SEGOFFSET, SEGSIZE, UPDSEGSIZE,
                    item_ct1,
                    dpct_local_acc_ct1.get_pointer());
            });
        });
    //CUT_CHECK_ERROR("upd_getReassigns_CUDA() kernel execution failed");

    // step 2: Determine offsets to move assignments of wrongly assigned data points to buffer 
    serialPrefixSum_KPSMCUDA(K, UPDSEGSIZE, UPDSEGOFFSET + 1);

    // step 3: Use string compaction to collect indices of wrongly assigned data points (moves assignments only)
    sMem = sizeof(int) * (5 * THREADSPERBLOCK + 4);
    q_ct1.submit([&](sycl::handler& cgh) {
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
                upd_moveAssigns_CUDA(N, ASSIGN, UPDASSIGN, SEGOFFSET,
                    UPDSEGOFFSET, item_ct1,
                    dpct_local_acc_ct1.get_pointer());
            });
        });
    //CUT_CHECK_ERROR("upd_moveAssigns_CUDA() kernel execution failed");

    // step 4: Updates number of data points per segment by checking wrongly assigned for new ones
    sMem = sizeof(int) * (THREADSPERBLOCK + 1);
    q_ct1.submit([&](sycl::handler& cgh) {
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
                upd_segmentSize_CUDA(UPDASSIGN, UPDSEGOFFSET + K, SEGSIZE,
                    item_ct1,
                    dpct_local_acc_ct1.get_pointer());
            });
        });
    //CUT_CHECK_ERROR("upd_segmentSize_CUDA() kernel execution failed");

    // step 5: Determine new global offsets
    serialPrefixSum_KPSMCUDA(K, SEGSIZE, SEGOFFSET + 1);

    // step 6: With new global offsets determine how many elements are in wrong segment
    sMem = sizeof(int) * (THREADSPERBLOCK + 2);
    q_ct1.submit([&](sycl::handler& cgh) {
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
                upd_findWrongAssigns_CUDA(N, ASSIGN, SEGOFFSET, UPDSEGSIZE,
                    item_ct1,
                    dpct_local_acc_ct1.get_pointer());
            });
        });
    //CUT_CHECK_ERROR("upd_findWrongAssigns_CUDA() kernel execution failed");

    // step 7: Check how many data points are wrongly assigned in total
    //         if that number >= N/2 then sort all
    //         if that number < 50% of N then update sort
    serialPrefixSum_KPSMCUDA(K, UPDSEGSIZE, UPDSEGOFFSET + 1);

    int* numReassigns = (int*)malloc(sizeof(int) * 1);
    q_ct1.memcpy(numReassigns, UPDSEGOFFSET + K, sizeof(int) * 1).wait();

    if (numReassigns[0] * 2 >= N)
    {
        // step 8a: Updating less efficient than full re-sort, return
        return false;
        // done
    }
    else
    {
        // step 8b: Write data to buffer
        sMem = sizeof(int) * (5 * THREADSPERBLOCK + 4);
        q_ct1.submit([&](sycl::handler& cgh) {
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
                    upd_moveDataToBuffer_CUDA(N, D, X, X2, ASSIGN, UPDASSIGN,
                        SEGOFFSET, UPDSEGOFFSET, item_ct1,
                        dpct_local_acc_ct1.get_pointer());
                });
            });
        //CUT_CHECK_ERROR("upd_moveDataToBuffer_CUDA() kernel execution failed");

        // step 9: Check data in buffer and write to data array to fill gaps in segments
        sMem = sizeof(int) * (8 * THREADSPERBLOCK);
        q_ct1.submit([&](sycl::handler& cgh) {
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
                    upd_collectDataFromBuffer_CUDA(
                        N, D, X, X2, ASSIGN, UPDASSIGN, SEGOFFSET,
                        UPDSEGOFFSET + K, item_ct1,
                        dpct_local_acc_ct1.get_pointer());
                });
            });
        //CUT_CHECK_ERROR("upd_collectDataFromBuffer_CUDA() kernel execution failed");
        return true;
        // done
    }
}


void sortData(int N, int K, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGSIZE, int* SEGOFFSET)
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
    sycl::range<3> block(THREADSPERBLOCK, 1,
        1);       // set number of threads per block
    sycl::range<3> gridK(K, 1, 1); // K blocks of threads in grid

    // loop over all data points, detect those that are assigned to cluster k,
    // determine unique memory indices in contiguous stretch of shared memory 
    // using string compaction with parallel prefix sum, then move data to buffer

    // run over all data points and detect those assigned to cluster K
    int sMem = (sizeof(int) * THREADSPERBLOCK);
    q_ct1.submit([&](sycl::handler& cgh) {
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
    //CUT_CHECK_ERROR("sort_getSegmentSize_CUDA() kernel execution failed");

    // first segment offset is 0 (as of initialization), compute the others as running sum over segment sizes
    serialPrefixSum_KPSMCUDA(K, SEGSIZE, SEGOFFSET + 1);

    // now move the data from X to X2
    sMem = sizeof(int) * (5 * THREADSPERBLOCK + 2);
    q_ct1.submit([&](sycl::handler& cgh) {
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
                sort_moveData_CUDA(N, D, X, X2, ASSIGN, UPDASSIGN, SEGOFFSET,
                    item_ct1, dpct_local_acc_ct1.get_pointer());
            });
        });
    //CUT_CHECK_ERROR("sort_moveData_CUDA() kernel execution failed");
}



FLOAT_TYPE kpsmeansGPU(int N, int K, int D, FLOAT_TYPE* x, FLOAT_TYPE* ctr, int* assign, unsigned int maxIter, DataIO* data)
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
    // CUDA kernel parameters
    sycl::range<3> block(THREADSPERBLOCK, 1, 1);
    sycl::range<3> gridK(K, 1, 1);
    sycl::range<3> gridN((int)ceil((FLOAT_TYPE)N / (FLOAT_TYPE)THREADSPERBLOCK),
        1, 1);
    int sMemAssign = (sizeof(FLOAT_TYPE) * THREADSPERBLOCK);
    int sMemScore = (sizeof(FLOAT_TYPE) * 2 * THREADSPERBLOCK);
    int sMemCenters = (sizeof(FLOAT_TYPE) * THREADSPERBLOCK + sizeof(int) * THREADSPERBLOCK);

    // GPU memory pointers, allocate and initialize device memory
    FLOAT_TYPE* x_d = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * N * D, x);
    FLOAT_TYPE* x2_d = data->allocDeviceMemory<float*>(sizeof(float) * N * D);
    FLOAT_TYPE* ctr_d = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * K * D, ctr);
    int* segsize_d = data->allocDeviceMemory<int*>(sizeof(int) * K);
    int* segoffs_d = data->allocZeroedDeviceMemory<int*>(sizeof(int) * (K + 1));
    int* upd_segsize_d = data->allocDeviceMemory<int*>(sizeof(int) * K);
    int* upd_segoffs_d = data->allocZeroedDeviceMemory<int*>(sizeof(int) * (K + 1));
    int* assign_d = data->allocDeviceMemory<int*>(sizeof(int) * N);
    int* upd_assign_d = data->allocDeviceMemory<int*>(sizeof(int) * N);
    FLOAT_TYPE* s_d = data->allocZeroedDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * K);

    // Initialize host memory
    FLOAT_TYPE* s = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * K);

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
                q_ct1.submit([&](sycl::handler& cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                        sycl::access::target::local>
                        dpct_local_acc_ct1(
                            sycl::range<1>(sMemCenters + 2 * sizeof(int)), cgh);

                    auto dpct_global_range = gridK * block;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(dpct_global_range.get(2),
                                dpct_global_range.get(1),
                                dpct_global_range.get(0)),
                            sycl::range<3>(block.get(2), block.get(1),
                                block.get(0))),
                        [=](sycl::nd_item<3> item_ct1) {
                            calcCentroidsSorted_CUDA(
                                N, D, x_d, ctr_d, segoffs_d, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                    });
                //CUT_CHECK_ERROR("calcCentroidsSorted_CUDA() kernel execution failed");
            }
            // for unsorted data
            else
            {
                q_ct1.submit([&](sycl::handler& cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                        sycl::access::target::local>
                        dpct_local_acc_ct1(sycl::range<1>(sMemCenters), cgh);

                    auto dpct_global_range = gridK * block;

                    cgh.parallel_for(
                        sycl::nd_range<3>(
                            sycl::range<3>(dpct_global_range.get(2),
                                dpct_global_range.get(1),
                                dpct_global_range.get(0)),
                            sycl::range<3>(block.get(2), block.get(1),
                                block.get(0))),
                        [=](sycl::nd_item<3> item_ct1) {
                            calcCentroids_CUDA(
                                N, D, x_d, ctr_d, assign_d, item_ct1,
                                dpct_local_acc_ct1.get_pointer());
                        });
                    });
                //CUT_CHECK_ERROR("calcCentroids_CUDA() kernel execution failed");
            }
            sorted = false;
        }
        iter++;

        // update clusters and create backup of current cluster centers
        q_ct1.submit([&](sycl::handler& cgh) {
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
                    assignToClusters_KPSMCUDA(N, K, D, x_d, ctr_d, assign_d,
                        item_ct1,
                        dpct_local_acc_ct1.get_pointer());
                });
            });
        //CUT_CHECK_ERROR("assignToClusters_KPSMCUDA() kernel execution failed");

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
            FLOAT_TYPE* temp_d = x_d;
            x_d = x2_d;
            x2_d = temp_d;

            // swap old assignments with new assignments
            int* a_temp_d = assign_d;
            assign_d = upd_assign_d;
            upd_assign_d = a_temp_d;

            sorted = true;
        }

        // for sorted data
        if (sorted)
        {
            // get score per cluster for sorted data
            q_ct1.submit([&](sycl::handler& cgh) {
                sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                    sycl::access::target::local>
                    dpct_local_acc_ct1(
                        sycl::range<1>(sMemScore + 2 * sizeof(int)), cgh);

                auto dpct_global_range = gridK * block;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                        dpct_global_range.get(1),
                        dpct_global_range.get(0)),
                        sycl::range<3>(block.get(2), block.get(1),
                            block.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        calcScoreSorted_CUDA(N, D, x_d, ctr_d, s_d, segoffs_d,
                            item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
                });
            //CUT_CHECK_ERROR("calcScoreSorted_CUDA() kernel execution failed");
        }
        // for unsorted data
        else
        {
            // get score per cluster for unsorted data
            q_ct1.submit([&](sycl::handler& cgh) {
                sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                    sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(sMemScore), cgh);

                auto dpct_global_range = gridK * block;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                        dpct_global_range.get(1),
                        dpct_global_range.get(0)),
                        sycl::range<3>(block.get(2), block.get(1),
                            block.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        calcScore_CUDA(N, D, x_d, ctr_d, assign_d, s_d,
                            item_ct1,
                            dpct_local_acc_ct1.get_pointer());
                    });
                });
            //CUT_CHECK_ERROR("calcScore_CUDA() kernel execution failed");
        }

        // copy scores per cluster back to the host and do reduction on CPU
        q_ct1.memcpy(s, s_d, sizeof(FLOAT_TYPE) * K).wait();
        score = 0.0;
        for (int i = 0; i < K; i++) score += s[i];
    }
    cout << "Number of iterations: " << iter << endl;

    // copy centroids back to host
    q_ct1.memcpy(ctr, ctr_d, sizeof(FLOAT_TYPE) * K * D).wait();
    // copy assignments back to host
    q_ct1.memcpy(assign, assign_d, sizeof(int) * N).wait();

    // free memory
    sycl::free(x_d, q_ct1);
    sycl::free(x2_d, q_ct1);
    sycl::free(ctr_d, q_ct1);
    sycl::free(segsize_d, q_ct1);
    sycl::free(segoffs_d, q_ct1);
    sycl::free(upd_segsize_d, q_ct1);
    sycl::free(upd_segoffs_d, q_ct1);
    sycl::free(assign_d, q_ct1);
    sycl::free(upd_assign_d, q_ct1);
    sycl::free(s_d, q_ct1);
    free(s);

    return score;
}
