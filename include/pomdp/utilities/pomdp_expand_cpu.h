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


#ifndef POMDP_EXPAND_CPU_H
#define POMDP_EXPAND_CPU_H


#include "pomdp.h"
#include "policies/pomdp_alpha_vectors.h"


/**
 *  Expand the set of beliefs following random trajectories (e.g., Perseus' expansion). This assumes that
 *  the variable B contains only one element: The initial belief b0. From this, B is expanded to the
 *  size specified; all are reachable belief points from random horizons. This assigns numDesiredBeliefPoints
 *  new elements to Bnew.
 *  @param  pomdp                   The POMDP object.
 *  @param  numDesiredBeliefPoints  The number of belief points desired after randomly adding beliefs.
 *  @param  maxNonZeroValues        The maximum number of non-zero values over all new belief points.
 *  @param  Bnew                    The new (raw) resultant belief points (numDesiredBeliefPoints-n array).
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_expand_random_cpu(const POMDP *pomdp, unsigned int numDesiredBeliefPoints,
        unsigned int *maxNonZeroValues, float *Bnew);

/**
 *  Expand the set of beliefs by selecting the most distinct successor belief possible for each belief
 *  in the current set B (e.g., PBVI's original expansion). This assigns pomdp->r new elements to Bnew.
 *  @param  pomdp               The POMDP object.
 *  @param  maxNonZeroValues    The maximum number of non-zero values over all new belief points.
 *  @param  Bnew                The new (raw) resultant belief points (r-n array).
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_expand_distinct_beliefs_cpu(const POMDP *pomdp, unsigned int *maxNonZeroValues, float *Bnew);

/**
 *  Expand the set of beliefs following the Point-based Error Minimization Algorithm (PEMA)
 *  variation of PBVI. This assigns one new element to Bnew.
 *  @param  pomdp               The POMDP object.
 *  @param  Rmin                The minimum reward.
 *  @param  Rmax                The maximum reward.
 *  @param  Gamma               The alpha-vectors from the solution for the current B (r-n array).
 *  @param  maxNonZeroValues    The maximum number of non-zero values over all new belief points.
 *  @param  Bnew                The new (raw) resultant belief points (r-n array).
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_expand_pema_cpu(const POMDP *pomdp, const POMDPAlphaVectors *policy,
        unsigned int *maxNonZeroValues, float *Bnew);


#endif // POMDP_EXPAND_CPU_H

