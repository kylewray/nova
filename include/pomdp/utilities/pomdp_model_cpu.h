/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2016 Kyle Hollins Wray, University of Massachusetts
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


#ifndef POMDP_UTILITIES_CPU_H
#define POMDP_UTILITIES_CPU_H


#include "pomdp.h"

namespace nova {

/**
 *  Perform a belief update.
 *  @param  pomdp       The POMDP object.
 *  @param  b           The current belief (n-array).
 *  @param  a           The action taken (index).
 *  @param  o           The observation made (index).
 *  @param  bp          The resulting new belief.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_belief_update_cpu(const POMDP *pomdp, const float *b,
        unsigned int a, unsigned int o, float *bp);

/**
 *  Free the memory for *only* the POMDP's internal arrays.
 *  @param  pomdp       The POMDP object. Arrays within will be freed.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_uninitialize_cpu(POMDP *pomdp);

};


#endif // POMDP_UTILITIES_CPU_H

