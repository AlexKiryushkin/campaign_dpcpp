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
 * \file kpsmeansGPU.cc
 * \brief Test/example file for presorted k-means clustering on the GPU
 *
 * Test/example file for presorted k-means clustering on the GPU
 *
 * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
 * \date 2/2/2010
 * \version 1.0
 **/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "../utils_dpcpp/defaults.h"
#include "../utils_dpcpp/dataio.h"
#include "../utils_dpcpp/gpudevices.h"
#include "kpsmeans_gpu.h"

//using namespace std;

  /**
   * \brief Main for testing
   */
int main(int argc, const char* argv[])
{
    Defaults* defaults = new Defaults(argc, argv, "kc");

    // get data and information about data from datafile
    DataIO* data = new DataIO;

    GpuDevices* systemGpuDevices = new GpuDevices(defaults);

    int i;
    FLOAT_TYPE score = 0.0f;

    FLOAT_TYPE* x = data->readData(defaults->getInputFileName().c_str()); // data points

    int N = data->getNumElements();
    int K = data->getNumClusters();
    int D = data->getDimensions();
    FLOAT_TYPE* ctr = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * K * D); // array containing centroids
    memset(ctr, 0, sizeof(FLOAT_TYPE) * K * D);
    int* assign = (int*)malloc(sizeof(int) * N); // assignments
    memset(assign, 0, sizeof(int) * N);

    // initialize first centroids with first K data points  
    // for each cluster
    for (i = 0; i < K; i++)
    {
        // for each dimension
        for (int d = 0; d < D; d++)
        {
            ctr[i * D + d] = x[d * N + i];
        }
    }

    // do clustering on GPU 
    score = kpsmeansGPU(N, K, D, x, ctr, assign, 0, data);

    // print results
    data->printClusters(N, K, D, x, ctr, assign);
    cout << "Score: " << score << endl;

    // free memory
    free(x);
    free(ctr);
    free(assign);

    // done
    cout << "Done clustering" << endl;

    return 0;
}