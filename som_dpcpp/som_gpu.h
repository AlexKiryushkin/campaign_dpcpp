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

/* $Id: somGPU.h 158 2011-01-12 23:06:27Z msosnick $ */
/**
 * \File somGPU.h
 * \brief A CUDA self organizing map implementation
 *
 * Implements self-organizing map (som) clustering on the GPU
 * 
 * \author Author: Kai Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 */


#ifndef HAVE_CONFIG_H
// defined globally in campaign.h
// define distance metric, e.g. CAMPAIGN_DISTANCE_MANHATTAN, CAMPAIGN_DISTANCE_EUCLIDEAN_, etc.
#define CAMPAIGN_DISTANCE_MANHATTAN /** < Type of distance metric */
#define THREADSPERBLOCK 256         /** < Threads per block (tpb) */
#define FLOAT_TYPE float            /** < Precision of floating point numbers */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <cmath>
#include <cfloat>
#include "../utils_dpcpp/dataio.h"
#include "../utils_dpcpp/timing.h"
#include "../utils_dpcpp/defaults.h"
#include "../utils_dpcpp/metrics_gpu.dp.hpp"
#include "../utils_dpcpp/gpudevices.h"

#else 
#include "../campaign.h"
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
static void reduceMinTwo(int tid, T *s_A, U *s_B, sycl::nd_item<3> item_ct1);


/**
 * \brief For a given input vector, finds the best matching unit among all weight vectors
 *
 * \param N Number of input vectors
 * \param K Number of weight vectors
 * \param D Number of dimensions
 * \param v Index of selected input vector
 * \param x Input vectors
 * \param wv Weight vectors
 * \param bmu Variable for returning bmu index (array size equals number of thread blocks)
 * \param dists Variable for returning bmu distances (one for each thread block)
 * \return Updated value of bmu and dists
 */
static void findBMU_CUDA(int N, int K, int D, int v, FLOAT_TYPE *X, FLOAT_TYPE *WV, int *BMU, FLOAT_TYPE *DISTS,
                         sycl::nd_item<3> item_ct1, uint8_t *dpct_local);

/**
 * \brief Creates a copy of BMU
 * \param K Number of weight vectors
 * \param D Number of dimensions
 * \param bmu Index of best matching vector
 * \param WV Weight vectors
 * \param BMU Best matching unit
 * \return Updated BMU array
 */
static void copyBMU_CUDA(int K, int D, int bmu, FLOAT_TYPE *WV, FLOAT_TYPE *BMU,
                         sycl::nd_item<3> item_ct1);



/**
 * \brief Updates all weight vectors given an input vector, the best matching unit, and a scaling factor
 * \param N Number of input vectors
 * \param K Number of weight vectors
 * \param D Number of dimensions
 * \param v Index of selected input vector
 * \param d0 Distance from bmu to input vector
 * \param BMU Best matching unit
 * \param X Input vectors
 * \param WV Weight vectors
 * \return Updated value of WV
 */
static void updateNeighborhood_CUDA(int N, int K, int D, int v, FLOAT_TYPE d0, FLOAT_TYPE scale, FLOAT_TYPE *BMU, FLOAT_TYPE *X, FLOAT_TYPE *WV,
                                    sycl::nd_item<3> item_ct1,
                                    uint8_t *dpct_local);


/**
 * \brief Runs self organizing map on the GPU. Requires a CUDA-enabled GPU
 *
 * \param N Number of input vectors
 * \param K Number of weight vectors
 * \param D Number of dimensions
 * \param numIter Number of iterations to be carried out
 * \param x Input vectors
 * \param pwv Reference to weight vectors
 * \return Updated weight vectors after numIter iterations
 */ 
void somGPU(int N, int K, int D, int numIter, FLOAT_TYPE *x, FLOAT_TYPE **pwv, DataIO *data);


