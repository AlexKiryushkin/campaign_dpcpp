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
  * \file gpudevices.cpp
  * \brief CUDA device detection and querying.
  *
  * CUDA device detection and querying.
  *
  * \author Author: Marc Sosnick, Contributors: Kai J. Kohlhoff, William Hsu
  * \date 5/26/10
  * \version pre-0.1
  */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gpudevices.h"

namespace  {

int getMaxGflopsDeviceId()
{
  int devicesCount{};
  int deviceId{};
  int maxGflops{};
  devicesCount = dpct::dev_mgr::instance().device_count();
  for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
  {
    dpct::device_info deviceProperties{};
    dpct::dev_mgr::instance()
        .get_device(deviceIndex)
        .get_device_info(deviceProperties);
    int currGflops = 2 * deviceProperties.get_max_clock_frequency() *
                     deviceProperties.get_max_compute_units();
    if (maxGflops < currGflops)
    {
      maxGflops = currGflops;
      deviceId = deviceIndex;
    }
  }
  return deviceId;
}

} // namespace

void GpuDevices::init() {

  if (GPUDEVICES_DEBUG) cout << "GpuDevices::init()" << endl;

  deviceCount = dpct::dev_mgr::instance().device_count();
  dpct::device_info *thisDevice;

  for (int i = 0; i < deviceCount; i++) {
    thisDevice = (dpct::device_info *)malloc(sizeof(dpct::device_info));
    dpct::dev_mgr::instance().get_device(i).get_device_info();
    if (thisDevice == NULL) {
      cout << "Error: cannot allocate memory for device information\n";
      exit(1);
    }
    else {
      properties.push_back(thisDevice);
    }
  }

  if (deviceCount == 1 && properties[0]->get_major_version() == 9999 && properties[0]->get_minor_version() == 9999) {
    cout << "Error: no CUDA capable device detected" << endl;
    exit(1);
  }
}

GpuDevices::GpuDevices() {
  init();

  setCurrentDevice(getMaxGflopsDeviceId());
}

GpuDevices::GpuDevices(bool detect) {
  init();

  if (detect) {
    setCurrentDevice(getMaxGflopsDeviceId());
  }
  else {
    setCurrentDevice(0);
  }
}

GpuDevices::GpuDevices(Defaults* defaults) {
  init();

  // user wants to display list of devices and exit
  if (defaults->getListDevices() == true) {
    cout << "GpuDevices: defaults getlistdevices" << endl;
    printDeviceList();
    exit(0);
  }

  // if getDetectDevices, user wants the fastest device on the system 
  // otherwise the user has specified the device to be used
  if (defaults->getDetectDevice() == true) {
    if (setCurrentDevice(getMaxGflopsDeviceId()) == GPUDEVICES_FAILURE) {
      cout << "Error: could not select device " << defaults->getDevice() << endl;
      exit(1);
    }
  }
  else {
    if (setCurrentDevice(defaults->getDevice()) == GPUDEVICES_FAILURE) {
      cout << "Error: could not select device " << defaults->getDevice() << endl;
      exit(1);
    }
  }
}

GpuDevices::GpuDevices(int major, int minor) {
  init();

  /*dpct::device_info deviceProp;

  deviceProp.set_major_version(major);
  deviceProp.set_minor_version(minor);
  int dev;
  cudaChooseDevice(&dev, &deviceProp);*/

  setCurrentDevice(0);
}


int GpuDevices::setCurrentDevice(int device) {

  if (device < deviceCount) {
    dpct::dev_mgr::instance().select_device(device);
    return GPUDEVICES_SUCCESS;
  }
  else {
    return GPUDEVICES_FAILURE;
  }
}

int GpuDevices::getCurrentDevice() {
  return currentDevice;
}


int GpuDevices::getDeviceCount() {
  return deviceCount;
}

dpct::device_info *GpuDevices::deviceProps(int device) {
  if (device < deviceCount) {
    return properties[device];
  }
  else {
    return NULL;
  }
}

void GpuDevices::printDeviceList() {

  for (int i = 0; i < deviceCount; i++) {
    cout << "CUDA-enabled devices present on this system: " << endl;
    cout << i << "  " << properties[i]->get_name() << endl;
  }
}

/*
 main(int argc, const char **argv){

 GpuDevices myDevices;

 cout << "Number of devices on system: " << myDevices.getDeviceCount() << endl;

 cout << "Device 0: " << myDevices.deviceProps(0)->name << endl;
 }
 */
