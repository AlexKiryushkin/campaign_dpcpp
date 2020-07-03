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
  * \file hierarchicalGPU.h
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
  // defined globally in campaign.h
  // define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_MANHATTAN /** < Type of distance metric */
#define THREADSPERBLOCK 256         /** < Threads per block (tpb) */
#define FLOAT_TYPE float            /** < Precision of floating point numbers */

#include <iostream>
#include <cfloat>
#include "../utils_cuda/dataio.h"
#include "../utils_cuda/timing.h"
#include "../utils_cuda/metricsGPU.cuh"
#include "../utils_cuda/defaults.h"
#include "../utils_cuda/gpudevices.h"

#endif

#undef _GLIBCXX_ATOMIC_BUILTINS

using namespace std;

/**
 * \brief Parallel algorithm, reduction (minimum of elements) of two arrays containing values and keys
 * Runtime O(log(tpb)) = O(1)
 * Works for up to 1024 elements in arrays
 * Called from within a kernel, will be inlined
 *
 * \param tid Thread ID
 * \param s_A Array of values in shared memory
 * \param s_B Array of keys in shared memory
 * \return Result of reduction in first elements of arrays s_A and s_B
 */
template <unsigned int BLOCKSIZE, class T, class U>
__device__ static void reduceMinTwo(int tid, T* s_A, U* s_B);


/**
 * \brief Calculates N * (N - 1) / 2 distances and stores nearest neighbor for all N data points
 * Slow algorithm for computing distance matrix, but with negligible contribution to total running time.
 *
 * \param N Number of data points
 * \param D Number of dimensions
 * \param x Clustering input data
 * \param NEIGHBOR List of indices of nearest neighbors
 * \param NEIGHBORDIST List of distances to nearest neighbors
 * \return Returns updated NEIGHBOR and NEIGHBORDIST
 */
__global__ static void calcDistanceMatrix_CUDA(int N, int D, FLOAT_TYPE* X, int* NEIGHBOR, FLOAT_TYPE* NEIGHBORDIST);

/**
 * \brief Parallel reduction to find minimum value in an array of arbitrary size
 * Iterative reduction algorithm, several iterative calls might be required.
 * Final iteration returns min value and key at first position of output arrays
 *
 * \param N Number of values in array
 * \param iter Number of iteration
 * \param INPUT N values
 * \param INKEY Keys for values
 * \param OUTPUT Result of reduction, one per block used
 * \param OUTKEY Keys for reduced values, one per block used
 * \return Minimum value and key pairs per block
 **/
__global__ void min_CUDA(unsigned int N, unsigned int iter, FLOAT_TYPE* INPUT, int* INKEY, FLOAT_TYPE* OUTPUT, int* OUTKEY);

/**
 * \brief Find index of cluster that is closest to its nearest neighbor
 * \param numClust Number of clusters
 * \param DISTS Precomputed neighbor heuristic distances
 * \param INDICES Precomputed neighbor heuristic indices
 * \param REDUCED_DISTS Returns list of per-block minimal distances
 * \param REDUCED_INDICES Returns list of indices for element that have per-block minimal distances
 * \return Returns index of cluster with minimum distance to its nearest neighbor
 */
int getMinDistIndex(int numClust, FLOAT_TYPE* DISTS, int* INDICES, FLOAT_TYPE* REDUCED_DISTS, int* REDUCED_INDICES);


/**
 * \brief Merges clusters at positions A and B and stores at position A
 * \param N Number of data points
 * \param D Number of dimensions
 * \param indexA Position of first cluster
 * \param indexB Position of second cluster
 * \param X Clustering input data
 * \return Updated list of positions X
 */
__global__ static void mergeElementsInsertAtA_CUDA(int N, int D, int indexA, int indexB, FLOAT_TYPE* X);


/**
 * \brief Computes all distances to element A
 * \param N Number of data points
 * \param numClust Number of clusters
 * \param D Number of dimensions
 * \param indexA Position of element A
 * \param X Clustering input data
 * \param DISTS Precomputed neighbor heuristic distances
 * \param INDICES Precomputed neighbor heuristic indices
 * \param REDUCED_DISTS Returns list of per-block minimal distances
 * \param REDUCED_INDICES Returns list of indices for element that have per-block minimal distances
 * \return Updated list of positions X and initial reduction of per-block minimal distances and corresponding indices
 * \TODO includes distance from old B to A, which will later be discarded
 */
__global__ static void computeAllDistancesToA_CUDA(int N, int numClust, int D, int indexA, FLOAT_TYPE* X, FLOAT_TYPE* DISTS, int* INDICES, FLOAT_TYPE* REDUCED_DISTS, int* REDUCED_INDICES);



/**
 * \brief Replaces element at position A with new element
 * \param indexA Position of the element that is to be replaced
 * \param DISTS Precomputed neighbor heuristic distances
 * \param INDICES Precomputed neighbor heuristic indices
 * \param REDUCED_DISTS Returns list of per-block minimal distances
 * \param REDUCED_INDICES Returns list of indices for element that have per-block minimal distances
 * \return Updated distances DISTS and indices INDICES with respect to element A
 */
__global__ static void updateElementA_CUDA(int indexA, FLOAT_TYPE* DISTS, int* INDICES, FLOAT_TYPE* REDUCED_DISTS, int* REDUCED_INDICES);


/**
 * \brief Look for nearest neighbor for element A by checking all elements at position < A
 * \param indexA Position of element A
 * \param sMem size of shared memory used by reduction algorithm
 * \param DISTS Precomputed neighbor heuristic distances
 * \param INDICES Precomputed neighbor heuristic indices
 * \param REDUCED_DISTS Returns list of per-block minimal distances
 * \param REDUCED_INDICES Returns list of indices for element that have per-block minimal distances
 * \return Updated distances DISTS and indices INDICES
 */
void updateDistanceAndIndexForCluster(int indexA, int sMem, FLOAT_TYPE* DISTS, int* INDICES, FLOAT_TYPE* REDUCED_DISTS, int* REDUCED_INDICES);


/**
 * \brief Moves cluster from position N to position B
 * \param N Number of clusters
 * \param D Number of dimensions
 * \param indexB Position to which to move
 * \param indexN Position from which to move
 * \param X Clustering input data
 * \param DISTS Precomputed neighbor heuristic distances
 * \param INDICES Precomputed neighbor heuristic indices
 * \return Updated list of positions X
 */
__global__ static void moveCluster_CUDA(int N, int D, int indexB, int indexN, FLOAT_TYPE* X, FLOAT_TYPE* DISTS, int* INDICES);


/**
 * \brief Computes all distances to element B for elements at positions following B
 * \param N Number of clusters
 * \param D Number of dimensions
 * \param indexB Position of element B
 * \param numElements Number of elements that require updating (typically number of existing clusters minus list position of element B)
 * \param X Clustering input data
 * \param DISTS Precomputed neighbor heuristic distances
 * \param INDICES Precomputed neighbor heuristic indices
 * \return Updated list of positions X, distances DISTS, and indices INDICES
 */
__global__ static void computeDistancesToBForPLargerThanB_CUDA(int N, int D, int indexB, int numElements, FLOAT_TYPE* X, FLOAT_TYPE* DISTS, int* INDICES);


/**
 * \brief Finds nearest neighbor of J in all elements in list before J
 * \param N Number of clusters
 * \param D Number of dimensions
 * \param indexJ Position of element J
 * \param X Clustering input data
 * \param DISTS Precomputed neighbor heuristic distances
 * \param INDICES Precomputed neighbor heuristic indices
 * \param REDUCED_DISTS Returns list of per-block minimal distances
 * \param REDUCED_INDICES Returns list of indices for element that have per-block minimal distances
 * \return Updated list of positions X and initial reduction of per-block minimal distances and corresponding indices
 */
__global__ static void recomputeMinDistanceForElementAt_j_CUDA(int N, int D, int indexJ, FLOAT_TYPE* X, FLOAT_TYPE* DISTS, int* INDICES, FLOAT_TYPE* REDUCED_DISTS, int* REDUCED_INDICES);


/**
 * \brief Performs hierarchical clustering on the GPU
 * \param N Number of data points
 * \param D Number of dimensions
 * \param x Clustering input data
 * \return List of paired indices in order of merging
 */
int* hierarchicalGPU(int N, int D, FLOAT_TYPE* x, DataIO* data);
