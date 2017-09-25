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


#ifndef NOVA_POMDP_EXPAND_CPU_H
#define NOVA_POMDP_EXPAND_CPU_H


#include <nova/pomdp/pomdp.h>
#include <nova/pomdp/policies/pomdp_alpha_vectors.h>

namespace nova {

/**
 *  Expand the set of beliefs following random trajectories (e.g., Perseus' expansion).
 *
 *  This randomly selects points from B, then randomly expands these points a random number of times.
 *  The process is repeated a number of times equal to the value provided. The new numBeliefsToAdd
 *  belief points are added to B and rz is updated accordingly.
 *
 *  @param  pomdp               The POMDP object. B and rz will be modified.
 *  @param  numBeliefsToAdd     The number of belief points to add.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_expand_random_cpu(POMDP *pomdp, unsigned int numBeliefsToAdd);

/**
 *  Expand the set of beliefs following random *unique* trajectories (e.g., Perseus' expansion).
 *
 *  This randomly selects points from B, then randomly expands these points a random number of times.
 *  The process is repeated a number of times equal to the value provided. The new numBeliefsToAdd
 *  belief points are added to B and rz is updated accordingly.
 *
 *  @param  pomdp               The POMDP object. B and rz will be modified.
 *  @param  numBeliefsToAdd     The number of belief points to add.
 *  @param  maxTrials           The maximum number of trials to perform while trying to find unique beliefs.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_expand_random_unique_cpu(POMDP *pomdp, unsigned int numBeliefsToAdd, unsigned int maxTrials);

/**
 *  Expand the set of beliefs by selecting the most distinct successor belief possible for each belief.
 *
 *  This is essentially PBVI's original expansion method. The new r belief points are added to B and
 *  rz is updated accordingly.
 *
 *  @param  pomdp               The POMDP object. B and rz will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_expand_distinct_beliefs_cpu(POMDP *pomdp);

/**
 *  Expand the set of beliefs following the Point-based Error Minimization Algorithm (PEMA).
 *
 *  This is for a variation of PBVI. The new single belief point is added to B and rz is updated accordingly.
 *
 *  @param  pomdp               The POMDP object. B and rz will be modified.
 *  @param  Gamma               The alpha-vectors from the solution for the current B (r-n array).
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_expand_pema_cpu(POMDP *pomdp, const POMDPAlphaVectors *policy);

};


#endif // NOVA_POMDP_EXPAND_CPU_H

