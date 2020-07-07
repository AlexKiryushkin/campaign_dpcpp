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
  * \File kmedoids_gpu.cc
  * \brief Test/example file for k-medoids clustering on the GPU
  *
  * Test/example file for k-medoids clustering on the GPU
  *
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 15/2/2010
  * \version 1.0
  **/

#ifdef HAVE_CONFIG_H
#include "../config.h"
#include "../campaign.h"
#else
#include "kmedoids_gpu.h"
#endif

using namespace std;


/**
 * \brief Main for testing
 */
int main(int argc, const char* argv[])
{
  // Initialization
  Timing timer;
  srand(2011); // fixed seed for reproducibility

  // Parse command line
  Defaults* defaults = new Defaults(argc, argv, "kmed");

  // select device based on command-line switches
  GpuDevices* systemGpuDevices = new GpuDevices(defaults);

  // get data and information about data from datafile
  DataIO* data = new DataIO;

  FLOAT_TYPE* x = data->readData(defaults->getInputFileName().c_str()); // data points

  int N = data->getNumElements();
  int K = data->getNumClusters();
  int D = data->getDimensions();
  int* medoid = (int*)malloc(sizeof(int) * K); // array with medoid indices
  int* assign = (int*)malloc(sizeof(int) * N); // assignments

  // initialize first set of medoids with first K data points  
  for (unsigned int k = 0; k < K; k++) medoid[k] = k;

  // do clustering on GPU 
  timer.start("kmedoidsGPU");
  FLOAT_TYPE score = kmedoidsGPU(N, K, D, x, medoid, assign, 100, data);
  timer.stop("kmedoidsGPU");

  // print results
  cout << "Final set of medoids: ";
  for (unsigned int k = 0; k < K; k++) cout << "[" << k << "] " << medoid[k] << "\t";
  cout << endl;
  cout << "Score: " << score << endl;

  if (defaults->getTimerOutput()) timer.report();

  // free memory
  free(x);
  free(medoid);
  free(assign);
  free(systemGpuDevices);

  // done
  cout << "Done clustering" << endl;

  return 0;
}