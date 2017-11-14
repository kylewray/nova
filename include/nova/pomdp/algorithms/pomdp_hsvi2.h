/**
 *  The MIT License (MIT)
 *
 *  Copyright (c) 2017 Kyle Hollins Wray, University of Massachusetts
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


#ifndef NOVA_POMDP_HSVI2_H
#define NOVA_POMDP_HSVI2_H


#include <nova/pomdp/pomdp.h>
#include <nova/pomdp/policies/pomdp_alpha_vectors.h>

namespace nova {

/**
 *  The necessary variables to perform heuristic search value iteration (v2) on a POMDP within nova.
 *  @param  trials                      The number of trials to try at a maximum.
 *  @param  epsilon                     The convergence criterion for the early termination of trials.
 *  @param  delta                       The *optional* convergence criterion for the change in width checked on each trial.
 *  @param  pruneGrowthThreshold        The percentage value (e.g., 0.1 = 10%) of growth after which alpha-vectors are pruned.
 *  @param  maxAlphaVectors             The maximum number of alpha-vectors <= trials * horizon.
 *  @param  currentTrial                The current trial updated in the outer loop of each iteration.
 *  @param  lowerGammaSize              The size of the lower Gamma alpha-vector arrays.
 *  @param  lowerGammaSizeLastPruned    The size of the lower Gamma alpha-vector arrays when the last pruning step was done.
 *  @param  lowerGamma                  The lower bound value of the beliefs as alpha-vectors (maxAlphaVectors-n-array).
 *  @param  lowerPi                     The lower bound action to take at each state (maxAlphaVectors-array).
 *  @param  upperGammaSize              The size of the upper Gamma point set arrays.
 *  @param  upperGammaSizeLastPruned    The size of the upper Gamma point set arrays when the last pruning step was done.
 *  @param  upperGammaB                 The upper bound beliefs in the point set (maxAlphaVectors-n-array).
 *  @param  upperGammaHVb               The upper bound value of the corresponding belief in the point set (maxAlphaVectors-array).
 */
typedef struct NovaPOMDPHSVI2 {
    unsigned int trials;
    float epsilon;
    float delta;
    float pruneGrowthThreshold;
    unsigned int maxAlphaVectors;

    unsigned int currentTrial;

    unsigned int lowerGammaSize;
    unsigned int lowerGammaSizeLastPruned;
    float *lowerGamma;
    unsigned int *lowerPi;

    unsigned int upperGammaSize;
    unsigned int upperGammaSizeLastPruned;
    float *upperGammaB;
    float *upperGammaHVb;
} POMDPHSVI2;

/**
 *  Execute all steps of HSVI2 for the infinite horizon POMDP model specified.
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2 object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_execute(const POMDP *pomdp, POMDPHSVI2 *hsvi2, POMDPAlphaVectors *policy);

/**
 *  Step 1/4: The initialization step of HSVI2. This sets up the Gamma and pi variables.
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2 object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_initialize(const POMDP *pomdp, POMDPHSVI2 *hsvi2);

/**
 *  Step 2/4: The update step of HSVI2. This applies the HSVI2 procedure once.
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2 object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_update(const POMDP *pomdp, POMDPHSVI2 *hsvi2);

/**
 *  Step 3/4: The get resultant policy step of HSVI2. This retrieves the alpha-vectors (Gamma) and
 *  corresponding actions (pi).
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2 object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_get_policy(const POMDP *pomdp, POMDPHSVI2 *hsvi2, POMDPAlphaVectors *policy);

/**
 *  Step 4/4: The uninitialization step of HSVI2. This sets up the Gamma and pi variables.
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2 object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_uninitialize(const POMDP *pomdp, POMDPHSVI2 *hsvi2);

}; // namespace nova

 
#endif // NOVA_POMDP_HSVI2_H


