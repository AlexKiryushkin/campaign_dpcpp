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
  * \File kmeans_gpu.h
  * \brief A basic CUDA K-means implementation
  *
  * A module of the CAMPAIGN data clustering library for parallel architectures
  * Implements parallel K-means clustering (base implementation) on the GPU
  *
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 12/2/2010
  * \version 1.0
  */

#ifndef HAVE_CONFIG_H
  // defined globally in campaign.h
  // define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED /** < Type of distance metric */
#define THREADSPERBLOCK 16 /** < Threads per block (tpb) */
#define FLOAT_TYPE float         /** < Precision of floating point numbers */

#include <iostream>
#include "../utils/dataio.h"
#include "../utils/timing.h"
#include "../utils/defaults.h"
#include "../utils/metrics_gpu.cuh"
#include "../utils/gpudevices.h"

#else
#include "../config.h"
#include "../campaign.h"
#endif

#define EPS 0.01            /** < Value of epsilon for convergence criteria */
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
__device__ static void reduceTwo(int tid, T* s_A, U* s_B);

/**
 * \brief Assign data points to clusters
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param X Clustering input data
 * \param CTR Centroid positions for next iteration
 * \param ASSIGN Assignments of data points to clusters
 * \return Updated values of ASSIGN
 */
__global__ static void assignToClusters_KMCUDA(int N, int K, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN);


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
__global__ static void calcScore_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN, FLOAT_TYPE* SCORE);


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
__global__ static void calcCentroids_CUDA(int N, int D, FLOAT_TYPE* X, FLOAT_TYPE* CTR, int* ASSIGN);


/**
 * \brief Runs K-means on theGPU. Requires CUDA-enabled graphics processor
 *
 * \param N Number of data points
 * \param K Number of clusters
 * \param D Number of dimensions
 * \param x Clustering input data
 * \param ctr Centroid positions
 * \param assign Assignments of data points to clusters
 * \param maxIter Maximum number of iterations
 * \param data pointer to DataIO object containing data to be clustered
 * \return Score for clustering after convergence and updated values of ctr and assign
 */
FLOAT_TYPE kmeansGPU(int N, int K, int D, FLOAT_TYPE* x, FLOAT_TYPE* ctr, int* assign, unsigned int maxIter, DataIO* data);
