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
  * \file rmsdGPU.cu
  * \brief A CUDA implementation of Theobald's algorithm
  *
  * An implementation of Theobald's algorithm for the calculation of
  * protein backbone C-alpha rmsds using quaternions for the GPU
  *
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 10/22/2009
  * \version 1.0
  */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "rmsd_gpu.h"
#include <cmath>

using namespace std;

static void centerAllConformations(int numAtoms, int numConf, FLOAT_TYPE* conf,
    sycl::nd_item<3> item_ct1)
{
    // get global thread ID
    int i, t = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
        item_ct1.get_local_id(2);
    // if active thread
    if (t < (3 * numConf))
    {
        int startID = t * numAtoms; // offset of coordinate c of conformation t
        // { x1...xN, y1...yN, z1...zN }
        FLOAT_TYPE sum = 0.0;
        // for each atom sum up coordinate
        for (i = 0; i < numAtoms; i++) sum += conf[startID + i];
        // if not already centered
        if (sum != 0.0)
        {
            // calculate center
            sum /= (FLOAT_TYPE)numAtoms;
            // transpose conformation along coordinate to origin
            for (i = 0; i < numAtoms; i++) conf[startID + i] -= sum;
        }
    }
}

static void getRMSD(int numConf, int numAtoms, int ctrConf, FLOAT_TYPE* conf, FLOAT_TYPE* rmsd,
    sycl::nd_item<3> item_ct1)
{
    // get global thread ID
    int t = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
        item_ct1.get_local_id(2);
    if (t < numConf)
    {
        int m = DIM;
        // ATTENTION: M resides in local mem, but is accessed frequently and not coalesced -> slow!
        FLOAT_TYPE M[9];
        for (int i = 0; i < m * m; i++) M[i] = 0;
        int index1 = t * numAtoms * m, index2 = ctrConf * numAtoms;

        // mult firstConfMatrix with referenceConfMatrix
        for (int i = 0; i < numAtoms; i++)
        {
            // non-coalesced memory reads
            // xA * xB
            M[0] += conf[index1 + i] * conf[index2 + i];
            // xA * yB
            M[1] += conf[index1 + i] * conf[index2 + numAtoms + i];
            // xA * zB
            M[2] += conf[index1 + i] * conf[index2 + 2 * numAtoms + i];
        }

        for (int i = 0; i < numAtoms; i++)
        {
            // yA * xB
            M[3] += conf[index1 + numAtoms + i] * conf[index2 + i];
            // yA * yB
            M[4] += conf[index1 + numAtoms + i] * conf[index2 + numAtoms + i];
            // yA * zB
            M[5] += conf[index1 + numAtoms + i] * conf[index2 + 2 * numAtoms + i];
        }

        for (int i = 0; i < numAtoms; i++)
        {
            // zA * xB
            M[6] += conf[index1 + 2 * numAtoms + i] * conf[index2 + i];
            // zA * yB
            M[7] += conf[index1 + 2 * numAtoms + i] * conf[index2 + numAtoms + i];
            // zA * zB
            M[8] += conf[index1 + 2 * numAtoms + i] * conf[index2 + 2 * numAtoms + i];
        }


        FLOAT_TYPE G_x = 0, G_y = 0;
        for (unsigned int i = 0; i < 3 * numAtoms; i++)
        {
            FLOAT_TYPE numA = conf[index2 + i];
            FLOAT_TYPE numB = conf[index1 + i];
            G_x += numA * numA;  // G_A
            G_y += numB * numB;  // G_B
        }

        // DEV: still too many glob mem accesses, can reduce this
        FLOAT_TYPE k00 = M[0 + 0 * m] + M[1 + 1 * m] + M[2 + 2 * m];      // [0, 0]
        FLOAT_TYPE k01 = M[1 + 2 * m] - M[2 + 1 * m];                 // [0, 1]
        FLOAT_TYPE k02 = M[2 + 0 * m] - M[0 + 2 * m];                 // [0, 2]
        FLOAT_TYPE k03 = M[0 + 1 * m] - M[1 + 0 * m];                 // [0, 3]
        FLOAT_TYPE k11 = M[0 + 0 * m] - M[1 + 1 * m] - M[2 + 2 * m];      // [1, 1]
        FLOAT_TYPE k12 = M[0 + 1 * m] + M[1 + 0 * m];                 // [1, 2]
        FLOAT_TYPE k13 = M[2 + 0 * m] + M[0 + 2 * m];                 // [1, 3]
        FLOAT_TYPE k22 = -M[0 + 0 * m] + M[1 + 1 * m] - M[2 + 2 * m];      // [2, 2]
        FLOAT_TYPE k23 = M[1 + 2 * m] + M[2 + 1 * m];                 // [2, 3]
        FLOAT_TYPE k33 = -M[0 + 0 * m] - M[1 + 1 * m] + M[2 + 2 * m];      // [3, 3]


        // float C_4 = 1.0, C_3 = 0.0;
        FLOAT_TYPE C_2, C_1, C_0;
        C_2 = 0;
        for (unsigned int i = 0; i < m * m; i++)
        {
            C_2 += M[i] * M[i];
        }
        C_2 *= -2;
        FLOAT_TYPE detM = 0.0, detK = 0.0;

        // get determinante M
        // could use rule of Sarrus, but better:
        // computationally more efficient with Laplace expansion
        detM = M[0] * (M[4] * M[8] - M[5] * M[7])
            + M[3] * (M[7] * M[2] - M[8] * M[1])
            + M[6] * (M[1] * M[5] - M[2] * M[4]);

        detK = k01 * k01 * k23 * k23 - k22 * k33 * k01 * k01 + 2 * k33 * k01 * k02 * k12
            - 2 * k01 * k02 * k13 * k23 - 2 * k01 * k03 * k12 * k23 + 2 * k22 * k01 * k03 * k13
            + k02 * k02 * k13 * k13 - k11 * k33 * k02 * k02 - 2 * k02 * k03 * k12 * k13
            + 2 * k11 * k02 * k03 * k23 + k03 * k03 * k12 * k12 - k11 * k22 * k03 * k03
            - k00 * k33 * k12 * k12 + 2 * k00 * k12 * k13 * k23 - k00 * k22 * k13 * k13
            - k00 * k11 * k23 * k23 + k00 * k11 * k22 * k33;


        C_1 = -8.0 * detM;
        C_0 = detK;

        FLOAT_TYPE lambda_old, lambda2, a, b;
        FLOAT_TYPE lambda = (G_x + G_y) / 2.0;
        unsigned int maxits = 50;
        FLOAT_TYPE tolerance = 1.0e-6;

        for (unsigned int i = 0; i < maxits; i++)
        {
            lambda_old = lambda;
            lambda2 = lambda_old * lambda_old;
            b = (lambda2 + C_2) * lambda_old;
            a = b + C_1;
            lambda = lambda_old - (a * lambda_old + C_0) / (2.0 * lambda2 * lambda_old + b + a);

            if (sycl::fabs(lambda - lambda_old) <
                sycl::fabs(tolerance * lambda)) break;
        }

        FLOAT_TYPE rmsd2 = (G_x + G_y - 2.0 * lambda) / numAtoms;
        FLOAT_TYPE ls_rmsd = 0.0;
        if (rmsd2 > 0)
            ls_rmsd = sycl::sqrt(rmsd2);

        rmsd[t] = ls_rmsd;
    }
}


FLOAT_TYPE* rmsdGPU(int numConf, int numAtoms, FLOAT_TYPE* conf, DataIO* data)
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
    const int threadsPerBlock = 3;
    sycl::range<3> block(threadsPerBlock, 1, 1); // X 1D threads per block
    // number of threads equivalent to number of conformations
    sycl::range<3> gridConf(
        (int)ceil((FLOAT_TYPE)numConf / (FLOAT_TYPE)threadsPerBlock), 1, 1);
    // number of conformations * 3: one thread for each coordinate of each conformation
    sycl::range<3> gridConf3(
        (int)ceil((FLOAT_TYPE)(3 * numConf) / (FLOAT_TYPE)threadsPerBlock), 1,
        1);

    // GPU memory allocation and initialization
    // allocate CONF on device and initialize with values in conf
    FLOAT_TYPE* CONF = data->allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * numConf * numAtoms * DIM, conf);
    // allocate RMSD on device and initialize with zeroes
    FLOAT_TYPE* RMSD = data->allocZeroedDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * numConf);

    FLOAT_TYPE* rmsds = (FLOAT_TYPE*)malloc(sizeof(FLOAT_TYPE) * numConf); // Holds the scores calculated by the threads.

    // center all conformations if requested
    if (!isCentered)
    {
        q_ct1.submit([&](sycl::handler& cgh) {
            auto dpct_global_range = gridConf3 * block;

            cgh.parallel_for(
                sycl::nd_range<3>(
                    sycl::range<3>(dpct_global_range.get(2),
                        dpct_global_range.get(1),
                        dpct_global_range.get(0)),
                    sycl::range<3>(block.get(2), block.get(1), block.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    centerAllConformations(numAtoms, numConf, CONF, item_ct1);
                });
            });
        //CUT_CHECK_ERROR("centerAllConformations() kernel execution failed");
    }

    // calculate rmsds to reference structure (third argument)
    q_ct1.submit([&](sycl::handler& cgh) {
        auto dpct_global_range = gridConf * block;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                    dpct_global_range.get(1),
                    dpct_global_range.get(0)),
                sycl::range<3>(block.get(2), block.get(1), block.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
                getRMSD(numConf, numAtoms, 0, CONF, RMSD, item_ct1);
            });
        });
    //CUT_CHECK_ERROR("getRMSD() kernel execution failed");

    // copy results back to host
    q_ct1.memcpy(rmsds, RMSD, sizeof(FLOAT_TYPE) * numConf).wait();

    // free memory
    sycl::free(RMSD, q_ct1);
    sycl::free(CONF, q_ct1);
    return rmsds;
}

