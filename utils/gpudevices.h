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
 * Authors: Marc Sosnick                                                       *
 * Contributors: Kai J. Kohlhoff, William Hsu                                  *
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
  * \file gpudevices.h
  * \brief Header file for CUDA-enabled device detection and querying.
  *
  * \author Author: Marc Sosnick, Contributors: Kai J. Kohlhoff, William Hsu
  * \date 5/26/10
  * \version pre-0.1
  */

#ifndef __GPUDEVICES_H__
#define __GPUDEVICES_H__

#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "defaults.h"
using namespace std;

#define GPUDEVICES_SUCCESS  0   /**< Return value upon success */
#define GPUDEVICES_FAILURE  1   /**< Return value upon error. */

/**
 *  \def GPUDEVICES_DEBUG
 *  \brief Sets diagnostic output for development
 *  If set to anything other than 0, diangnostic messages
 *  will be output to stdio during runtime.
 */
#define GPUDEVICES_DEBUG 1

 /**
  * \brief Provides CUDA-enabled device detection and querying.
  *
  * GpuDevices detects all CUDA-enabled devices on the system, collects
  * information about them, and allows the user to query basic data
  * without having to analyze the cudaDeviceProp structure.
  *
  * \author Marc Sosnick
  * \date 5/26/10
  */
class           GpuDevices {
public:
  /**
   * \var vector<cudaDeviceProp *> properties
   * \brief Vector of pointers to CUDA property struct for each detected device.
   */
  vector < cudaDeviceProp* >properties;

  /**
   * \var int currentDevice
   * \brief Current CUDA device in use.
   */
  int             currentDevice;

  /**
   * \var int deviceCount
   * \brief The number of CUDA-enabled devices detected in the system
   */
  int             deviceCount;

  /**
   * \brief Automatically detects fastest CUDA enabled device in system
   * Uses the CUDA's cutGetMaxGflopsDeviceId to select device with maximum
   * flops on system.
   */
  GpuDevices();
  ~GpuDevices();

  /**
   * \brief If detect is false, chooses first device detected, otherwise chooses
   *        fastest device in system.
   * If detect is false, the first CUDA-enabled device detected in the system is
   * chosen.  If detect is true, the fastest device as determined by CUDA's
   * cutGetMaxGflopsDeviceId is selected.
   * \param detect true chooses fastest device in system.  If false, chooses first
   *        device detected.
   */
  GpuDevices(bool detect);


  /**
   * \brief Uses settings in defaults to determine device configuration
   * \param defaults pointer to a defaults object which contains information from the command line
   *  for the campaign system.
   */
  GpuDevices(Defaults* defaults);



  /**
   * \brief Chooses system that most closely matches major and
   *        minor compute capability.
   * By specifying a major and minor compute capability, the device most
   * closely matching these specifications is chosen by using
   * CUDA's cudaChooseDevice.
   */
  GpuDevices(int major, int minor);

  /**
   * \brief Sets the current device
   * \param device new current device
   * \return GPUDEVICES_SUCCESS if device set, otherwise GPUDEVICES_FAILURE.
   */
  int             setCurrentDevice(int device);

  /**
   * \brief Returns the number of the current CUDA device
   * \return int the number of the current CUDA device
   */
  int             getCurrentDevice();


  /**
   * \brief Returns the count of CUDA-enabled devices in the system
   * \return int the count of CUDA-enabled devices in the system.
   */
  int             getDeviceCount();

  /**
   * \brief Returns pointer to the requested device information struct.
   * \param device device number
   * \return cudaDeviceProp* If device is valid, a pointer to the requested device's information.
   *         If error, NULL.
   */
  cudaDeviceProp* deviceProps(int device);

  /**
   * \brief Outputs list of all CUDA-enabled devices to console
   */
  void            printDeviceList();

private:
  /**
   * \brief Detects any CUDA-enabled cards on
   * system, gets information about devices.
   */
  void            init();
};

#endif
