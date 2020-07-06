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
  * \file min.h
  * \brief Minimum value
  *
  * This is an iterative reduction that may have to be called repeatedly with
  * final iteration returning min value and thread index at first position of
  * output arrays. Reduces number of elements from N to one per block in each
  * iteration, and to one at final iteration.
  */

  /**
   * \param g_idata Array containing N values for which the min is to
   *        be determined
   * \param g_idataINT Array containing N indeces which correspond to thread
   *        IDs at first iteration
   * \param g_odata Array containing min values per block at first # block
   *        positions
   * \param g_odataINT Array containing indeces of min values
   * \param iter Iteration
   * \param N Number of values to reduce
   * \return Updated values of g_odata and g_odataINT
   **/
__global__ void min(float* g_idata, int* g_idataINT, float* g_odata, int* g_odataINT, unsigned int iter, unsigned int N);
