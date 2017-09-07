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


#ifndef NOVA_POMDP_HSVI2_CPU_H
#define NOVA_POMDP_HSVI2_CPU_H


#include <nova/pomdp/pomdp.h>
#include <nova/pomdp/policies/pomdp_alpha_vectors.h>

namespace nova {

/**
 *  The necessary variables to perform heuristic search value iteration (v2) on a POMDP within nova.
 *  @param  GammaInitial    The initial values for alpha-vectors (r-n-array).
 *  @param  currentHorizon  The current horizon updated after each iteration.
 *  @param  Gamma           The value of the states (n-array).
 *  @param  GammaPrime      The value of the states (n-array) copy.
 *  @param  pi              The action to take at each state (n-array).
 */
typedef struct NovaPOMDPHSVI2CPU {
    float *GammaInitial;

    unsigned int currentHorizon;

    float *Gamma;
    float *GammaPrime;
    unsigned int *pi;
} POMDPHSVI2CPU;

/**
 *  Step 1/3: The initialization step of HSVI2. This sets up the Gamma and pi variables.
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2CPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_initialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2);

/**
 *  Step 2/3: Execute HSVI2 for the infinite horizon POMDP model specified.
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2CPU object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_execute_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, POMDPAlphaVectors *policy);

/**
 *  Step 3/3: The uninitialization step of HSVI2. This sets up the Gamma and pi variables.
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2CPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_uninitialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2);

/**
 *  The update step of HSVI2. This applies the HSVI2 procedure once.
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2CPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_update_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2);

/**
 *  The get resultant policy step of HSVI2. This retrieves the alpha-vectors (Gamma) and
 *  corresponding actions (pi).
 *  @param  pomdp       The POMDP object.
 *  @param  hsvi2       The POMDPHSVI2CPU object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_hsvi2_get_policy_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, POMDPAlphaVectors *policy);

};

 
#endif // NOVA_POMDP_HSVI2_CPU_H


