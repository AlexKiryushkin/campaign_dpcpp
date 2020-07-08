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
  * \file rmsdGPU.cc
  * \brief Test/example file for clustering on the GPU
  *
  * Test/example file for hierarchical clustering on the GPU
  *
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 2/2/2010
  * \version 1.0
  */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "../utils_dpcpp/defaults.h"
#include "../utils_dpcpp/dataio.h"
#include "../utils_dpcpp/gpudevices.h"
#include "rmsd_gpu.h"

  /**
   * \brief Main for testing
   */
int main(int argc, const char* argv[])
{
    Defaults* defaults = new Defaults(argc, argv, "kc");

    GpuDevices* systemGpuDevices = new GpuDevices(defaults);
    systemGpuDevices->setCurrentDevice(0);

    const int seed = 0;
    DataIO* data = new DataIO;
    FLOAT_TYPE* x = data->readData(defaults->getInputFileName().c_str());

    // ND: for RMSD initial project is not valid due to incorrect reading from file
    // With 5, 10 and 3 in the beginning of the input file, only 5*3=15 values are read,
    //  so now file is changed for correct reading, but hardcodes set bellow! 
    int N = /*data->getNumElements()*/5;
    int K = /*data->getNumClusters()*/10; // stands for number of atoms in rmsdGPU
    int D = /*data->getDimensions()*/3;  // expected to be 3
    cout << "Reading " << N << " structures of " << K << " atoms each ("
        << D << " dimensions)" << endl;
    data->setDataSize(N * K * D);

    // do rmsd calculations on GPU
    FLOAT_TYPE* rmsd = rmsdGPU(N, K, x, data);

    // print resulting RMSDs
    cout << "RMSDS: " << endl;
    for (int i = 0; i < N; i++) cout << rmsd[i] << "\t";
    cout << endl;

    free(x);
    free(rmsd);

    cout << "Done clustering" << endl;

    return 0;
}