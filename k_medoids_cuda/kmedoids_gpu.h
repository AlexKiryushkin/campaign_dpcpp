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
  * \File kmedoids_gpu.h
  * \brief A CUDA K-medoids implementation with all-prefix sorting
  *
  * A module of the CAMPAIGN data clustering library for parallel architectures
  * Implements parallel K-medoids clustering with parallel
  * all-prefix sum sorting for the GPU
  *
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 15/2/2010
  * \version 1.0
  **/

#ifndef HAVE_CONFIG_H
  // defined globally in campaign.h
  // define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED     /** < Type of distance metric */
#define THREADSPERBLOCK 256             /** < Threads per block (tpb) */
#define FLOAT_TYPE float                /** < Precision of floating point numbers */


#include <iostream>
#include <ctime>
#include "../utils/dataio.h"
#include "../utils/timing.h"
#include "../utils/defaults.h"
#include "../utils/metrics_gpu.cuh"
#include "../utils/gpudevices.h"
#include "../utils/dataio.h"

// #include "../testing.h"
#else 
#include "../campaign.h"
#endif

#undef _GLIBCXX_ATOMIC_BUILTINS

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
__device__ static void reduceOne(int tid, T* s_A);


/**
 * \brief Assign data points to clusters
 * Runtime O(D*K*N)
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Medoid positions for next iteration
 * \param ASSIGN Assignments of data points to clusters
 * \return Updated values of ASSIGN
 */
__global__ static void assignToClusters_KMDCUDA(int N, int K, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN);


/**
 * \brief Compute score for current assignment using data sorted by assignment
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Medoid positions
 * \param SCORE Array to store score components for each block
 * \param SEGOFFSET K+1 segment boundaries for sorted data array X
 * \return Updated values of SCORE
 */
__global__ static void calcScoreSorted_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, FLOAT_TYPE* SCORE, int* SEGOFFSET);

/**
 * \brief Compute score for current assignment using data sorted by assignment
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Medoid positions
 * \param CTR2 Temporary storage for new medoid positions
 * \param INDEX Indices for data points in initial array
 * \param SCORE Array to store score components for each block
 * \param SEGOFFSET K+1 segment boundaries for sorted data array X
 * \param MEDOID Indices of medoids
 * \param RANDOM One random number per cluster to pick next medoids
 * \return Updated values of SCORE
 */
__global__ static void calcNewScoreSortedAndSwap_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, FLOAT_TYPE* CTR2, int* INDEX, FLOAT_TYPE* SCORE, int* SEGOFFSET, int* MEDOID, int* RANDOM);


// ************ SECTION FOR PARALLEL DATA SORTING *******************  

/**
 * \brief Parallel prefix sum (scan) according to GPU Gems Ch. 39
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
__device__ static int parallelPrefixSum(int tid, int* DATA);


/**
 * \brief Determines number of data points assigned to each cluster
 * Runtime O(K*N)
 *
 * \param N Number of data points
 * \param ASSIGN Assignments of data points to clusters
 * \param SEGSIZE Number of assignments to each cluster
 * \return Segment sizes in SEGSIZE
 */
__global__ static void sort_getSegmentSize_CUDA(int N, int* ASSIGN, int* SEGSIZE);



/**
 * \brief Move data from unsorted array to buffer
 * Runtime O(D*N)
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param X2 Buffer for clustering data sorted by assignment
 * \param INDEX Indices to locate data points in initial array
 * \param INDEX2 Buffer for rearranged data point indices after sorting
 * \param ASSIGN Assignments of data points to clusters
 * \param SEGOFFSET K+1 segment boundaries for sorted data in buffer
 * \return New sorted data points in X2 and sorted assignments in UPDASSIGN
 */
__global__ static void sort_moveData_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* INDEX, int* INDEX2, int* ASSIGN, int* SEGOFFSET);


/**
 * \brief Determines how many data points are assigned to each cluster
 *
 * \param N Size of array
 * \param INPUT Input array of size N in GPU global memory
 * \param OUTPUT Output array of size N + 1 in GPU global memory
 * \return Running sum over INPUT in OUTPUT
 */
 // Attention: CPU version
void serialPrefixSum_KMDCUDA(int N, int* INPUT, int* OUTPUT);


/**
 * \brief Sorts array of data points based on assignments using K blocks
 * Runtime O(D*K*N)
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param X2 Buffer for clustering data sorted by assignment
 * \param INDEX Indices to locate data points in initial array
 * \param INDEX2 Buffer for rearranged data point indices after sorting
 * \param ASSIGN Assignments of data points to clusters
 * \param SEGSIZE Number of assignments to each cluster
 * \param SEGOFFSET K+1 segment boundaries for data array
 * \return New sorted data points in X2, rearranged indices in INDEX2, data segment sizes in SEGSIZE, and segment boundaries in SEGOFFSET
 */
void sortData(int N, int K, int D, FLOAT_TYPE* X, FLOAT_TYPE* X2, int* INDEX, int* INDEX2, int* ASSIGN, int* SEGSIZE, int* SEGOFFSET);

/**
 * \brief Runs K-medoids on theGPU. Requires CUDA-enabled graphics processor
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param x Clustering input data
 * \param medoid Initial set of medoids
 * \param assign Assignments of data points to clusters
 * \param maxIter Maximum number of iterations
 * \return Score for assignment, new set of medoids in medoid and assignments in assign
 */
FLOAT_TYPE kmedoidsGPU(int N, int K, int D, FLOAT_TYPE* x, int* medoid, int* assign, unsigned int maxIter, DataIO* data);