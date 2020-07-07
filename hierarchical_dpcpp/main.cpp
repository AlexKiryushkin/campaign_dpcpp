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
  * \file hierarchical_gpu.cc
  * \brief Test/example file for hierarchical clustering on the GPU
  *
  * Test/example file for hierarchical clustering on the GPU
  *
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 2/2/2010
  * \version 1.0
  **/


#ifdef HAVE_CONFIG_H
#include "../config.h"
#include "../campaign.h"
#else
#include "hierarchical_gpu.h"
#endif


  /**
   * \brief Main for testing
   */
int main(int argc, const char* argv[])
{
  // Initialization
  Timing timer;

  // Parse command line
  Defaults* defaults = new Defaults(argc, argv, "hier");

  // select CUDA device based on command-line switches
  GpuDevices* systemGpuDevices = new GpuDevices(defaults);

  // get data and information about data from datafile
  DataIO* data = new DataIO;

  FLOAT_TYPE* x = data->readData(defaults->getInputFileName().c_str());

  int N = data->getNumElements();
  int K = data->getNumClusters();
  int D = data->getDimensions();

  // do clustering on GPU 
  timer.start("hierarchicalGPU");
  int* seq = hierarchicalGPU(N, D, x, data);
  timer.stop("hierarchicalGPU");

  free(x);
  // retrace clustering 
  x = data->readData(defaults->getInputFileName().c_str());

  int* ids = (int*)malloc(sizeof(int) * N);
  for (int i = 0; i < N; i++) ids[i] = i;

  // print all clusters (ATTENTION: only for D=1 and D=2)
  if (D < 3)
  {
    FLOAT_TYPE* num1 = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * D);
    FLOAT_TYPE* num2 = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * D);
    unsigned int pos1, pos2, id1, id2, nextID = N - 1;
    for (int i = 0; i < N - 1; i++)
    {
      nextID++;
      id1 = seq[2 * i]; id2 = seq[2 * i + 1];
      for (int j = 0; j < N; j++)
      {
        if (ids[j] == id1)
        {
          for (unsigned int d = 0; d < D; d++) num1[d] = x[j + d * N];
          pos1 = j;
        }
        if (ids[j] == id2)
        {
          for (unsigned int d = 0; d < D; d++) num2[d] = x[j + d * N];
          pos2 = j;
        }
      }
      cout << id1 << "\t" << id2 << "\t(";
      for (unsigned int d = 0; d < D; d++) cout << num1[d] << (d + 1 < D ? ", " : ") & (");
      for (unsigned int d = 0; d < D; d++) cout << num2[d] << (d + 1 < D ? ", " : ") => (");
      for (unsigned int d = 0; d < D; d++) cout << (num1[d] + num2[d]) / 2.0f << (d + 1 < D ? ", " : ") : ");
      cout << nextID << endl;
      for (unsigned int d = 0; d < D; d++)
      {
        x[d * N + pos1] = (x[d * N + pos1] + x[d * N + pos2]) / 2.0f;
      }
      ids[pos1] = nextID;
    }
  }

  if (defaults->getTimerOutput()) timer.report();

  // free memory
  free(data);
  free(x);
  free(systemGpuDevices);

  // done
  cout << "Done clustering" << endl;

  return 0;
}
