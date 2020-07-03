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
  * \file timing.cpp
  * \brief Named timer functionality
  *
  * Provides multiple simultaneous named timers.
  *
  * \author Author: Marc Sosnick, Contributors: Kai J. Kohlhoff, William Hsu
  * \date 5/26/10
  * \version pre-0.1
  */

#include "timing.h"

Timing::Timing() {
}

Timing::~Timing() {
}

void Timing::start(string timerName) {
  cudaEventCreate(&startMap[timerName]);
  cudaEventRecord(startMap[timerName], 0);
}


void Timing::stop(string timerName) {
  cudaEventCreate(&stopMap[timerName]);
  cudaEventRecord(stopMap[timerName], 0);
}


void Timing::report() {
  cudaEvent_t currentTime;
  cudaEventCreate(&currentTime);
  cudaEventRecord(currentTime, 0);
  float timeMs;
  string status = "";

  cout << "Current Timings:" << endl;
  cout << setw(15) << "Timer" << setw(15) << "Time (ms)" << setw(15) << "Status" << endl;
  for (map<string, cudaEvent_t>::iterator it = startMap.begin(); it != startMap.end(); ++it) {
    if (stopMap.find((*it).first) != stopMap.end()) {
      cudaEventElapsedTime(&timeMs, (*it).second, stopMap[(*it).first]);
      status = "done";
    }
    else {
      cudaEventElapsedTime(&timeMs, (*it).second, currentTime);
      status = "running";
    }

    cout << setw(15) << (*it).first << setw(15) << timeMs << setw(15) << status << endl;
  }
}

void Timing::report(string timerName) {
  cudaEvent_t currentTime;
  cudaEventCreate(&currentTime);
  cudaEventRecord(currentTime, 0);
  float timeMs;

  if (startMap.find(timerName) == startMap.end()) {
    cout << "Timer \"" << timerName << "\" was never started." << endl;
    return;
  }
  else if (stopMap.find(timerName) == stopMap.end()) {
    cudaEventElapsedTime(&timeMs, startMap[timerName], currentTime);
    cout << timerName << " = " << timeMs << " ms (running)" << endl;
    return;
  }
  cudaEventElapsedTime(&timeMs, startMap[timerName], stopMap[timerName]);
  cout << timerName << " = " << timeMs << " ms" << endl;
}


double TimingCPU::getTimeInMicroseconds(void) {
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto millis = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  double t = static_cast<double>(millis) / 1000.0;
  return t;
}


void TimingCPU::start(string timerName) {
  startMap[timerName] = getTimeInMicroseconds();
}


void TimingCPU::stop(string timerName) {
  stopMap[timerName] = getTimeInMicroseconds();
}


void TimingCPU::report() {
  double timeMs;
  string status = "";

  cout << "Current Timings:" << endl;
  cout << setw(15) << "Timer" << setw(15) << "Time (ms)" << setw(15) << "Status" << endl;
  for (map<string, double>::iterator it = startMap.begin(); it != startMap.end(); ++it) {
    if (stopMap.find((*it).first) != stopMap.end()) {
      timeMs = stopMap[(*it).first] - (*it).second;
      status = "done";
    }
    else {
      timeMs = getTimeInMicroseconds() - (*it).second;
      status = "running";
    }

    cout << setw(15) << (*it).first << setw(15) << (timeMs / 1000.0f) << setw(15) << status << endl;
  }
}

void TimingCPU::report(string timerName) {
  double timeMs;

  if (startMap.find(timerName) == startMap.end()) {
    cout << "Timer \"" << timerName << "\" was never started." << endl;
    return;
  }
  else if (stopMap.find(timerName) == stopMap.end()) {
    timeMs = getTimeInMicroseconds() - startMap[timerName];
    cout << timerName << " = " << timeMs << " ms (running)" << endl;
    return;
  }
  timeMs = stopMap[timerName] - startMap[timerName];
  cout << timerName << " = " << timeMs << " ms" << endl;
}
