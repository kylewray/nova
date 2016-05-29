/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2015 Kyle Hollins Wray, University of Massachusetts
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of
 *  this software and associated documentation files (the "Software"), to deal in
 *  the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *  the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *  FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 *  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 *  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 *  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#ifndef POMDP_SIGMA_CPU_H
#define POMDP_SIGMA_CPU_H


#include "pomdp.h"

#include <utility>

namespace nova {

/**
 *  Perform the sigma-approximation on the current set of beliefs and return a new set.
 *  @param  pomdp                       The POMDP object.
 *  @param  numDesiredNonZeroValues     The desired maximum number of non-zero values in belief vectors.
 *  @param  Znew                        The new set of state indexes after the approximation (r-rz array).
 *                                      This will be created and modified.
 *  @param  Bnew                        The new set of belief points after the approximation (r-rz array).
 *                                      This will be created and modified.
 *  @param  sigma                       The resultant sigma value, proportional to the approximation error.
 *                                      This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_sigma_cpu(POMDP *pomdp, unsigned int numDesiredNonZeroValues,
        int *&Znew, float *&Bnew, float &sigma);

// A quick typedef for comparing beliefs and remembering their indexes.
typedef std::pair<float, int> SigmaPair;

/**
 *  A comparator function for SigmaPair types.
 *  @param  bl  The left belief to compare.
 *  @param  br  The right belief to compare.
 *  @return Returns true if left is greater than right, false otherwise.
 */
bool pomdp_sigma_pair_comparator_cpu(const SigmaPair &bl, const SigmaPair &br);

};


#endif // POMDP_SIGMA_CPU_H

