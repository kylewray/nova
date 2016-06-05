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


#ifndef NOVA_POMDP_PERSEUS_CPU_H
#define NOVA_POMDP_PERSEUS_CPU_H


#include <nova/pomdp/pomdp.h>
#include <nova/pomdp/policies/pomdp_alpha_vectors.h>

namespace nova {

/**
 *  The necessary variables to perform Perseus on a POMDP within nova.
 *  @param  GammaInitial    The initial values for alpha-vectors (r-n-array).
 *  @param  currentHorizon  The current horizon updated after each iteration.
 *  @param  rGamma          The actual number of belief points s.t. rGamma <= r.
 *  @param  rGammaPrime     The actual number of belief points s.t. rGammaPrime <= r.
 *  @param  rTilde          The number of beliefs that still exist in the set BTilde.
 *  @param  BTilde          The beliefs that still have not improved in value after an update.
 *  @param  Gamma           The value of the states (n-array).
 *  @param  GammaPrime      The value of the states (n-array) copy.
 *  @param  pi              The action to take at each state (n-array).
 *  @param  pi              The action to take at each state (n-array) copy.
 */
typedef struct NovaPOMDPPerseusCPU {
    float *GammaInitial;

    unsigned int currentHorizon;

    unsigned int rGamma;
    unsigned int rGammaPrime;

    unsigned int rTilde;
    unsigned int *BTilde;

    float *Gamma;
    float *GammaPrime;
    unsigned int *pi;
    unsigned int *piPrime;
} POMDPPerseusCPU;

/**
 *  Step 1/3: The initialization step of Perseus. This sets up the Gamma and pi variables.
 *  @param  pomdp       The POMDP object.
 *  @param  per         The POMDPPerseusCPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_initialize_cpu(const POMDP *pomdp, POMDPPerseusCPU *per);

/**
 *  Step 2/3: Execute Perseus for the infinite horizon POMDP model specified.
 *  @param  pomdp       The POMDP object.
 *  @param  per         The POMDPPerseusCPU object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_execute_cpu(const POMDP *pomdp, POMDPPerseusCPU *per,
        POMDPAlphaVectors *policy);

/**
 *  Step 3/3: The uninitialization step of Perseus. This sets up the Gamma and pi variables.
 *  @param  pomdp       The POMDP object.
 *  @param  per         The POMDPPerseusCPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_uninitialize_cpu(const POMDP *pomdp, POMDPPerseusCPU *per);

/**
 *  The update step of Perseus. This applies the Perseus procedure once.
 *  @param  pomdp       The POMDP object.
 *  @param  per         The POMDPPerseusCPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_update_cpu(const POMDP *pomdp, POMDPPerseusCPU *per);

/**
 *  The get resultant policy step of Perseus. This retrieves the alpha-vectors (Gamma) and
 *  corresponding actions (pi).
 *  @param  pomdp       The POMDP object.
 *  @param  per         The POMDPPerseusCPU object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_perseus_get_policy_cpu(const POMDP *pomdp, POMDPPerseusCPU *per,
        POMDPAlphaVectors *policy);

};


#endif // NOVA_POMDP_PERSEUS_CPU_H

