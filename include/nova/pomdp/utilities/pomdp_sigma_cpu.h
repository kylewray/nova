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


#ifndef NOVA_POMDP_SIGMA_CPU_H
#define NOVA_POMDP_SIGMA_CPU_H


#include <nova/pomdp/pomdp.h>

#include <utility>
#include <vector>

namespace nova {

/**
 *  Perform the sigma-approximation on the current set of beliefs and revise them within the POMDP.
 *  @param  pomdp                       The POMDP object.
 *  @param  numDesiredNonZeroValues     The desired maximum number of non-zero values in belief vectors.
 *  @param  sigma                       The resultant sigma value, proportional to the approximation error.
 *                                      This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_sigma_cpu(POMDP *pomdp, unsigned int numDesiredNonZeroValues, float &sigma);

};


#endif // NOVA_POMDP_SIGMA_CPU_H

