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
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 12/2/2010
  * \version 1.0
  **/

#ifdef HAVE_CONFIG_H
#include "../config.h"
#include "../campaign.h"
#else
  // defined globally in campaign.h
  // define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED /** < Type of distance metric */
#define THREADSPERBLOCK 128 /** < Threads per block (tpb) */
#define FLOAT_TYPE float         /** < Precision of floating point numbers */

#undef _GLIBCXX_ATOMIC_BUILTINS

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <cfloat>
#include "../utils_dpcpp/dataio.h"
#include "../utils_dpcpp/defaults.h"
#include "../utils_dpcpp/metrics_gpu.dp.hpp"
#endif

#define EPS 0.01            /** < Value of epsilon for convergence criteria */
#define DOSORTING true      /** < Enable/disable sorting stage */
#define DOUPDATING true     /** < Enable/disable updating stage (if sorting enabled) */

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
static void reduceOne(int tid, T* s_A, sycl::nd_item<3> item_ct1);


/**
 * \brief Parallel algorithm, reduction (sum of elements) of two arrays congruently
 * Runtime O(log(tpb)) = O(1)
 * Works for up to 1024 elements in arrays
 * Called from within a kernel, will be inlined
 *
 * \param tid Thread ID
 * \param s_A Array in shared memory
 * \param s_B Array in shared memory
 * \return Result of reduction in first elements of arrays s_A and s_B
 */
template <unsigned int BLOCKSIZE, class T, class U>
static void reduceTwo(int tid, T* s_A, U* s_B, sycl::nd_item<3> item_ct1);


/**
 * \brief Assign data points to clusters
 * Runtime O(D*K*N)
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Centroid positions for next iteration
 * \param ASSIGN Assignments of data points to clusters
 * \return Updated values of ASSIGN
 */
static void assignToClusters_KPSMCUDA(int N, int K, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local);



/**
 * \brief Compute score for current assignment using data sorted by assignment
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Centroid positions
 * \param SCORE Array to store score components for each block
 * \param SEGOFFSET K+1 segment boundaries for sorted data array X
 * \return Updated values of SCORE
 */
static void calcScoreSorted_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, FLOAT_TYPE* SCORE, int* SEGOFFSET,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local);


/**
 * \brief Computes score for current assignment
 * Runtime O(D*K*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Centroid positions
 * \param ASSIGN Assignments of data points to clusters
 * \param SCORE Array to store score components for each block
 * \return Updated values of SCORE
 */
static void calcScore_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN, FLOAT_TYPE* SCORE,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local);

/**
 * \brief Compute new centroids
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Cluster center positions for next iteration
 * \param SEGOFFSET K+1 segment boundaries for sorted data array X
 * \return Updated values of CTR

 */
static void calcCentroidsSorted_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* SEGOFFSET,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local);


/**
 * \brief Compute new centroids
 * Runtime O(D*K*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Centroid positions
 * \param ASSIGN Assignments of data points to clusters
 * \return Updated values of CTR
 */
static void calcCentroids_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local);


// ************ SECTION FOR PARALLEL DATA SORTING *******************  

/**
 * \brief Parallel prefix sum (scan) according to GPU Gems Ch. 39.
 * Part of Parallel data sorting.
 * Runtime O(log(tpb)) = O(1)
 * Works for up to 1024 elements in array
 * Uses modified parallel prefix sum
 * Called from within a kernel, will be inlined
 *
 * \param tid Thread ID
 * \param DATA Shared memory used for intermediate and final results
 * \return Running sum in DATA (permuted)
 */
template <unsigned int BLOCKSIZE>
static int parallelPrefixSum(int tid, int* DATA, sycl::nd_item<3> item_ct1);


/**
 * \brief Determines number of data points assigned to each cluster.
 * Part of Parallel data sorting.
 * Runtime O(K*N)
 *
 * \param N Number of data points
 * \param ASSIGN Assignments of data points to clusters
 * \param SEGSIZE Number of assignments to each cluster
 * \return Segment sizes in SEGSIZE
 */
static void sort_getSegmentSize_CUDA(int N, int* ASSIGN, int* SEGSIZE,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local);


/**
 * \brief Move data from unsorted array to buffer.
 * Part of Parallel data sorting.
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param X2 Buffer for clustering data sorted by assignment
 * \param ASSIGN Assignments of data points to clusters
 * \param UPDASSIGN Buffer for sorted assignments
 * \param SEGOFFSET K+1 segment boundaries for sorted data in buffer
 * \return New sorted data points in X2 and sorted assignments in UPDASSIGN
 */
static void sort_moveData_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGOFFSET,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local);


/**
 * \brief Determines how many data points are assigned to each cluster
 * Part of Parallel data sorting.
 *
 * \param N Size of array
 * \param INPUT Input array of size N in GPU global memory
 * \param OUTPUT Output array of size N + 1 in GPU global memory
 * \return Running sum over INPUT in OUTPUT
 */
 // Attention: CPU version
void serialPrefixSum_KPSMCUDA(int N, int* INPUT, int* OUTPUT);



// ************ SECTION FOR PARALLEL SORT UPDATING  *******************

/**
 * \brief Determines how many data points are mis-assigned in each segment.
 * Part of parallel sort updating.
 * Runtime O(N)
 *
 * \param N Number of data points
 * \param ASSIGN Assignments of data points to clusters
 * \param SEGOFFSET K+1 segment boundaries for sorted data
 * \param SEGSIZE Number of assignments to each cluster
 * \param UPDSEGSIZE Number of wrongly assigned data points in segments
 * \return Number of wrongly assigned data points in segments in UPDSEGSIZE andelements in SEGSIZE decremented by that value
 */
static void upd_getReassigns_CUDA(int N, int* ASSIGN, int* SEGOFFSET, int* SEGSIZE, int* UPDSEGSIZE,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local);


/**
 * \brief Move assigns from unsorted array to buffer
 * Part of parallel sort updating.
 * Runtime O(N)
 *
 * \param N Number of data points
 * \param ASSIGN Assignments of data points to clusters
 * \param UPDASSIGN Buffer for assignments
 * \param SEGOFFSET K+1 segment boundaries for last sorted array
 * \param UPDSEGOFFSET K+1 segment boundaries in buffer
 * \return Buffer UPDASSIGN with assignments of reassigned data points
 */
static void upd_moveAssigns_CUDA(int N, int* ASSIGN, int* UPDASSIGN, int* SEGOFFSET, int* UPDSEGOFFSET,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local);


/**
 * \brief Searches buffer to determine how many data points are assigned to each cluster
 * Part of parallel sort updating.
 * Runtime O(K*N)
 *
 * \param UPDASSIGN Buffer with assignments for data points
 * \param BUFFERSIZE Size of buffer
 * \param SEGSIZE Number of assignments to each cluster
 * \return Updated segment sizes in SEGSIZE
 */
static void upd_segmentSize_CUDA(int* UPDASSIGN, int* BUFFERSIZE, int* SEGSIZE,
    sycl::nd_item<3> item_ct1, uint8_t* dpct_local);


/**
 * \brief Determines how many data points are assigned to each cluster
 * Part of parallel sort updating.
 * Runtime O(K*N)
 *
 * \param N Number of data points
 * \param ASSIGN Assignments of data points to clusters
 * \param SEGOFFSET K+1 segment boundaries for sorted data
 * \param UPDSEGSIZE Number of wrongly assigned data points in segments
 * \return Number of wrongly assigned data points in UPDSEGSIZE
 */
static void upd_findWrongAssigns_CUDA(int N, int* ASSIGN, int* SEGOFFSET, int* UPDSEGSIZE,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local);

/**
 * \brief Move incorrectly assigned data points from unsorted array to buffer
 * Part of parallel sort updating.
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param X2 Buffer for clustering data
 * \param ASSIGN Assignments of data points to clusters
 * \param UPDASSIGN Buffer for assignments
 * \param SEGOFFSET K+1 segment boundaries for data array
 * \param UPDSEGOFFSET K+1 segment boundaries for buffer
 * \return Moved data points in X2 and their assignments in UPDASSIGN
 */
static void upd_moveDataToBuffer_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGOFFSET, int* UPDSEGOFFSET,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local);


/**
 * \brief Collect data from buffer and write into available positions in data array
 * Part of parallel sort updating.
 * Runtime O(K*N+D*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param X2 Buffer for clustering data
 * \param ASSIGN Assignments of data points to clusters
 * \param UPDASSIGN Buffer with assignments for data points
 * \param SEGOFFSET K+1 segment boundaries for data array
 * \param BUFFERSIZE Size of buffer
 * \return New sorted data points in X2
 */
static void upd_collectDataFromBuffer_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGOFFSET, int* BUFFERSIZE,
    sycl::nd_item<3> item_ct1,
    uint8_t* dpct_local);


/**
 * \brief Reestablishes sorted data array after subset was reassigned
 * Part of parallel sort updating.
 * Runtime O(D*K*N)
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param X2 Buffer for clustering data
 * \param ASSIGN Assignments of data points to clusters
 * \param UPDASSIGN Buffer for assignments
 * \param SEGSIZE Number of assignments to each cluster for previously sorted array
 * \param UPDSEGSIZE Storage for buffer segment sizes
 * \param SEGOFFSET K+1 segment boundaries for data array
 * \param UPDSEGOFFSET Storage for K+1 segment boundaries in buffer
 * \return If true, new sorted data points in X, assignments in ASSIGN, new segment sizes in SEGSIZE and new segment offsets in SEGOFFSET. If false, updating failed
 */
bool updateData(int N, int K, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGSIZE, int* UPDSEGSIZE, int* SEGOFFSET, int* UPDSEGOFFSET);



/**
 * \brief Sorts array of data points based on assignments using K blocks
 * Part of parallel sort updating.
 * Runtime O(D*K*N)
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param X2 Buffer for clustering data sorted by assignment
 * \param ASSIGN Assignments of data points to clusters
 * \param UPDASSIGN Buffer for sorted assignments
 * \param SEGSIZE Number of assignments to each cluster
 * \param SEGOFFSET K+1 segment boundaries for data array
 * \return New sorted data points in X2, sorted assignments in UPDASSIGN, data segment sizes in SEGSIZE, and segment boundaries in SEGOFFSET
 */
void sortData(int N, int K, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* ASSIGN, int* UPDASSIGN, int* SEGSIZE, int* SEGOFFSET);

/**
 * \brief Runs K-means on the GPU. Requires CUDA-enabled graphics processor
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param x Clustering input data
 * \param ctr Centroid positions
 * \param assign Assignments of data points to clusters
 * \param maxIter Maximum number of iterations
 * \param data DataIO object with information about data
 * \return Score for clustering after convergence and updated values of ctr and assign
 */
FLOAT_TYPE kpsmeansGPU(int N, int K, int D, FLOAT_TYPE* x, FLOAT_TYPE* ctr, int* assign, unsigned int maxIter, DataIO* data);
