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
 * \File somGPU.cu
 * \brief A CUDA self organizing map implementation
 *
 * Implements som clustering
 * 
 * \author Kai Kohlhoff
 * \date 2/2/2010
 * \version 1.0
 */


#ifdef HAVE_CONFIG_H
#include "../config.h"
#include "../campaign.h"
#else
#include "som_gpu.h"
#endif

/**
 * \brief Main for testing
 */
int main(int argc, const char* argv[])
{
    // Initialization
    Timing timer;
    
    // Parse command line
    Defaults *defaults = new Defaults(argc, argv, "som");
    
    // select device based on command-line switches
    GpuDevices *systemGpuDevices = new GpuDevices(defaults);
    
    // get data and information about data from datafile
    DataIO* data = new DataIO;
    
    FLOAT_TYPE* x = data->readData(defaults->getInputFileName().c_str());
    int N = data->getNumElements();  // number of input vectors
    int D = data->getDimensions();   // dimensions of input and weight vectors
    int K2D = 10;                    // number of vectors per dimension
    int K = K2D * K2D;               // number of weight vectors
    int numIter = 1;                 // number of iterations
    FLOAT_TYPE* wv = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * K * D); // array of weight vectors
    memset(wv, 0, sizeof(FLOAT_TYPE) * K * D);
    
    // initialize weight vectors by finding min and max in every dimension
    // and spacing vector coordinates evenly between those extremes
    for (unsigned int d = 0; d < D; d++)
    {
        // get limits for dimension d
        FLOAT_TYPE min = x[0];
        FLOAT_TYPE max = x[0];
        for (unsigned int i = d * N; i < (d + 1) * N; i ++)
        {
            if (x[i] > max) max = x[i];
            if (x[i] < min) min = x[i];
        }
        cout << "max: " << max << endl;
        cout << "min: " << min << endl;
        
        // update learning restraint
        FLOAT_TYPE increment = (max - min) / (FLOAT_TYPE) (K2D - 1);
        
        // initialize map 
        for (unsigned int i = 0; i < K2D; i++)
        {
            for (unsigned int j = 0; j < K2D; j++)
            {
                if (d == 0) wv[    i * K2D + j] = min + i * increment;
                else        wv[K + j * K2D + i] = min + i * increment;
            }
        }
    }
    
    // do clustering on GPU 
    timer.start("somGPU");
    somGPU(N, K, D, numIter, x, &wv, data); // wv has to be changeable
    timer.stop("somGPU");
    
    // print results  
    for (unsigned int i = 0; i < K; i++)
    {
        for (int d = 0; d < D; d++)
            cout << wv[d * K + i] << "\t";
        cout << endl;
    }
    cout << endl;
    if (defaults->getTimerOutput()) timer.report();
    
    // free memory
    free(x);
    free(wv);
    
    // done  
    cout << "Done clustering" << endl;
    
    return 0;
}

