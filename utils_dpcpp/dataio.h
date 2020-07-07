#pragma once

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
  * \file dataio.h
  * \brief Header for file and stream I/O routines
  *
  * \author Author: Kai J. Kohlhoff, Contributors: Marc Sosnick, William Hsu
  * \date 2/2/2010
  * \version 1.0
  **/

#ifndef __DATAIO_H_
#define __DATAIO_H_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

  /**
   * Support class for DataIO
   *
   * Support class for DataIO
   **/
class Internal;


/**
 * Implements a number of I/O routines for files and streams.
 *
 * DataIO provides data handling functionality.  DataIO provides data input and
 * output functionality, as well as some device memory handling and console output
 * routines.
 *
 **/
class DataIO
{
public:
  DataIO();
  ~DataIO();
  float* readData(const char* fileName);
  float* getData(); // read data from file or input stream
  const char* getFileName(); // get name of data file
  int getNumElements(); // get number of data points in set
  int getNumClusters(); // get number of clusters
  int getDimensions(); // get number of dimensions
  int getDataSize(); // get number of elements in data file
  void setDataSize(int numData); // set number of elements in data file
  void printClusters(int numData, int numClust, int numDim, float* data, float* ctr, int* assign); // print list of clusters

  /* TEMPLATES */

  /**
   * \brief Allocates device memory and initializes it with zeroes
   * \param memSize Size of memory to be allocated in bytes
   * \return Memory pointer
   */
  void * allocZeroedDeviceMemory(int memSize)
  {
    void * retVal;
    retVal = (void *)sycl::malloc_device(memSize, dpct::get_current_device(),
                                         dpct::get_default_context());
    dpct::get_default_queue().memset(retVal, 0, memSize).wait();
    return retVal;
  }


  /**
   * \brief Allocates device memory and initializes it with zeroes
   * \param memSize Size of memory to be allocated in bytes
   * \return Memory pointer
   */
  void * allocInitializedDeviceMemory(int memSize, int preSet)
  {
    void * retVal;
    retVal = (void *)sycl::malloc_device(memSize, dpct::get_current_device(),
                                         dpct::get_default_context());
    dpct::get_default_queue().memset(retVal, preSet, memSize).wait();
    return retVal;
  }

  /**
   * \brief Allocates device memory and copies values from host memory into it
   * \param memSize Size of memory to be allocated in bytes
   * \param data Host data of size memSize to be copied to allocated memory
   * \return Memory pointer
   */
  void * allocDeviceMemory(int memSize, void * data)
  {
    void * retVal;
    retVal = (void *)sycl::malloc_device(memSize, dpct::get_current_device(),
                                         dpct::get_default_context());
    dpct::get_default_queue()
        .memcpy(retVal, data, memSize)
        .wait(); // copy data from host to device
    return retVal;
  }

  /**
   * \brief Allocates uninitialized device memory
   * \param memSize Size of memory to be allocated in bytes
   * \return Memory pointer
   */
  void * allocDeviceMemory(int memSize)
  {
    void * retVal;
    retVal = (void *)sycl::malloc_device(memSize, dpct::get_current_device(),
                                         dpct::get_default_context());
    return retVal;
  }

  /**
     * \brief Allocates uninitialized device memory
     * \param memSize Size of memory to be allocated in bytes
     * \return Memory pointer
     */
  template <typename T>
  T allocDeviceMemory(int memSize)
  {
      T retVal;
      retVal = (T)sycl::malloc_device(memSize, dpct::get_default_queue());
      return retVal;
  }

  /**
     * \brief Allocates device memory and copies values from host memory into it
     * \param memSize Size of memory to be allocated in bytes
     * \param data Host data of size memSize to be copied to allocated memory
     * \return Memory pointer
     */
  template <typename T>
  T allocDeviceMemory(int memSize, T data)
  {
      dpct::device_ext& dev_ct1 = dpct::get_current_device();
      sycl::queue& q_ct1 = dev_ct1.default_queue();
      T retVal;
      retVal = (T)sycl::malloc_device(memSize, q_ct1);
      q_ct1.memcpy(retVal, data, memSize)
          .wait(); // copy data from host to device
      return retVal;
  }

  /**
   * \brief Allocates device memory and initializes it with zeroes
   * \param memSize Size of memory to be allocated in bytes
   * \return Memory pointer
   */
  template <typename T>
  T allocZeroedDeviceMemory(int memSize)
  {
      dpct::device_ext& dev_ct1 = dpct::get_current_device();
      sycl::queue& q_ct1 = dev_ct1.default_queue();
      T retVal;
      retVal = (T)sycl::malloc_device(memSize, q_ct1);
      q_ct1.memset(retVal, 0, memSize).wait();
      return retVal;
  }


private:
  Internal* ip;
};

#endif
