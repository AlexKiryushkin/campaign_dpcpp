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
 * \file min.cu
 * \brief Minimum value
 *
 * This is an iterative reduction that may have to be called repeatedly with 
 * final iteration returning min value and thread index at first position of 
 * output arrays. Reduces number of elements from N to one per block in each 
 * iteration, and to one at final iteration.
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "min.h"

void min(float *g_idata, int *g_idataINT, float *g_odata, int *g_odataINT, unsigned int iter, unsigned int N,
         sycl::nd_item<3> item_ct1,
         uint8_t *dpct_local)
{
  auto * array = (char*)  dpct_local;
  auto * sdata = (float*) array;
  auto * idata = (int*)  &sdata[item_ct1.get_local_range(2)];

  unsigned int tid = item_ct1.get_local_id(2);
  unsigned int i = item_ct1.get_global_id(2);
  sdata[tid] = 0;

  if (iter == 0) // Note: all threads follow the same branch
  {
    if (i < N)
    {
      sdata[tid] = g_idata[i];
      idata[tid] = i;
    }
  }
  else
  {
    idata[tid] = 0;
    if (i < N)
    {
      sdata[tid] = g_idata[i];
      idata[tid] = g_idataINT[i];
    }
  }
  item_ct1.barrier();

  for (unsigned int s = item_ct1.get_local_range(2) / 2; s > 0; s >>= 1)
  {
    if ((tid < s) && (i + s < N))
    {
      if (sdata[tid] == sdata[tid + s])
      {
        idata[tid] = sycl::min(idata[tid], idata[tid + s]);
      }
      else if (sdata[tid] > sdata[tid + s])
      {
        sdata[tid] = sdata[tid + s];
        idata[tid] = idata[tid + s];
      }
    }
    item_ct1.barrier();
  }

  // write result for this block to global mem
  if (tid == 0)
  {
    g_odata[item_ct1.get_group(2)] = sdata[tid];
    g_odataINT[item_ct1.get_group(2)] = idata[tid];
  }
}

