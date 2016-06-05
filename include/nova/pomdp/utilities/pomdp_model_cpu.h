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
 *  Allocate memory for *only* the POMDP's internal arrays, given the relevant parameters.
 *  @param  pomdp       The POMDP object. Only arrays within will be freed.
 *  @param  n           The number of states.
 *  @param  ns          The maximum number of successor states.
 *  @param  m           The number of actions.
 *  @param  z           The number of observations.
 *  @param  r           The number of belief points.
 *  @param  rz          The maximum number of non zero values over all beliefs.
 *  @param  gamma       The discount factor between 0.0 and 1.0.
 *  @param  horizon     The horizon of the MDP.
 *  @param  epsilon     The convergence criterion for algorithms like LAO*.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_initialize_cpu(POMDP *pomdp, unsigned int n, unsigned int ns, unsigned int m,
    unsigned int z, unsigned int r, unsigned int rz, float gamma, unsigned int horizon);

/**
 *  Perform a belief update.
 *  @param  pomdp       The POMDP object.
 *  @param  b           The current belief (n-array).
 *  @param  a           The action taken (index).
 *  @param  o           The observation made (index).
 *  @param  bp          The resulting new belief. This will be created and modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_belief_update_cpu(const POMDP *pomdp, const float *b,
        unsigned int a, unsigned int o, float *&bp);

/**
 *  Given a raw set of beliefs (numBeliefsToAdd-n array), this adds all Bnew elements into B.
 *  @param  pomdp               The POMDP object. The Z and B values will be freed, created, and modified.
 *  @param  numBeliefsToAdd     The number of beliefs in Bnew to add to B.
 *  @param  Bnew                The new raw beliefs to add to B (numBeliefsToAdd-n array).
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_add_new_raw_beliefs_cpu(POMDP *pomdp, unsigned int numBeliefsToAdd,
        float *Bnew);

/**
 *  Free the memory for *only* the POMDP's internal arrays.
 *  @param  pomdp       The POMDP object. Only arrays within will be freed.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_uninitialize_cpu(POMDP *pomdp);

};


#endif // POMDP_UTILITIES_CPU_H

