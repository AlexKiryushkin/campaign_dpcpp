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
  * \file timing.h
  * \brief Header file for named timer functionality
  *
  * Provides multiple simultaneous named timers.
  *
  * \author Author: Marc Sosnick, Contributors: Kai J. Kohlhoff, William Hsu
  * \date 5/26/10
  * \version 1.0
  */

#ifndef __TIMING_H_
#define __TIMING_H_

#include <chrono>
#include <map>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>

using namespace std;

/**
 * \brief Provides named CUDA timer functionality.
 *
 * The timing class allows the user to refer to CUDA timing
 * events by name.  Multiple timers can be started and stopped
 * by using this name.  Reports include the time as well as the
 * name of the timer.
 *
 * \section Example
 * The following example creates two timers, and times two functions.  After
 * timing the first function, the time for the first timer is reported.  After
 * the second timer is run, all timer results are reported.
 * \code
 * #include "timing.h"
 *
 * Timing myTimer;
 *
 * myTimer.start("demo timer 1");
 * functionToBeTimed();
 * myTimer.stop("demo timer 1");
 * myTimer.report("demo timer 1");
 * myTimer.start("demo timer 2");
 * anotherFunctionToBeTimed();
 * myTimer.stop("demo timer 2");
 * myTimer.report();
 * \endcode
 *
 * \author Marc Sosnick
 * \date 5/26/10
 */
class Timing {


private:
  map<string, cudaEvent_t> startMap; /**< CUDA events recorded at start time */
  map<string, cudaEvent_t> stopMap; /**< CUDA events recorded at stop time */

public:
  Timing();
  ~Timing();

  /**
   * \brief Starts a GPU timer.
   * \param timerName string name of the timer to start
   */
  void start(string timerName);

  /**
   * \brief Stops a GPU timer.
   * \param timerName string name of the timer to stop
   */
  void stop(string timerName);

  /**
   * \brief Reports state of all timers to stdout.
   */
  void report();

  /**
   * \brief Reports state of specified timer to stdout.
   * \param timerName string name of timer to report
   */
  void report(string timerName);
};

/*!
 * \brief Provides named CPU timer functionality.
 *
 * The timing class allows the user to refer to CPU timing
 * events by name.  Multiple timers can be started and stopped
 * by using a name.  Reports include the time as well as the
 * name of the timer.
 *
 * \author Kai Kohlhoff
 * \date 5/26/10
 */
class TimingCPU {

private:
  map<string, double> startMap; /**< Time stamp recorded at start time */
  map<string, double> stopMap;  /**< Time stamp recorded at stop time */

  /**
   * \brief gets time in microseconds
   * \return time in microseconds
   */
  double getTimeInMicroseconds(void);

public:
  /**
   * \brief starts a CPU timer
   * \param timerName String containing the name of the CPU timer to start.
   */
  void start(string timerName);

  /**
   * \brief stops a CPU timer
   * \param timerName String containing the name of the CPU timer to stop.
   */
  void stop(string timerName);

  /**
   * \brief Reports state of all CPU timers to stdout.
   */
  void report();

  /**
   * \brief Reports state of specified CPU timer to stdout.
   * \param timerName string name of CPU timer to report
   */
  void report(string);

};



#endif
