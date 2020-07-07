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
  * \file rmsdGPU.h
  * \brief A CUDA implementation of Theobald's algorithm
  *
  * An implementation of Theobald's algorithm for the calculation of
  * protein backbone C-alpha rmsds using quaternions for the GPU
  *
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 10/22/2009
  * \version 1.0
  */

#ifndef HAVE_CONFIG_H

#include <iostream>
#include "../utils/dataio.h"
#include "../utils/timing.h"
#include "../utils/gpudevices.h"
#include "../utils/defaults.h"

#define FLOAT_TYPE float
#else 
#include "../config.h"
#include "../campaign.h"
#endif

#define DIM 3
#define isCentered false

using namespace std;
/**
 * \brief calculates center of mass of C-alphas and shifts all structures to origin
 * each thread works on one coordinate of one conformation (3 * numConf threads)
 *
 * \param numAtoms number of atoms in each conformation (DIM coordinates each)
 * \param numConf  number of conformations in data set
 * \param conf coordinates of atoms in all conformations
 */
__global__ static void centerAllConformations(int numAtoms, int numConf, FLOAT_TYPE* conf);


/**
 * \brief get rmsd for a set of two structures
 */
__global__ static void getRMSD(int numConf, int numAtoms, int ctrConf, FLOAT_TYPE* conf, FLOAT_TYPE* rmsd);



/**
 * \brief Runs RMSD on the GPU.  Requires CUDA-enabled grapnics processor.
 *
 * \param numAtoms number of atoms in each conformation (DIM coordinates each)
 * \param numConf number of conformations in data set
 * \param conf coordinates of atoms in all conformations
 */
FLOAT_TYPE* rmsdGPU(int numConf, int numAtoms, FLOAT_TYPE* conf, DataIO* data);
