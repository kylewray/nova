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


#ifndef NOVA_POMDP_PBVI_H
#define NOVA_POMDP_PBVI_H


#include <nova/pomdp/pomdp.h>
#include <nova/pomdp/policies/pomdp_alpha_vectors.h>

namespace nova {

/**
 *  The necessary variables to perform point-based value iteration on a POMDP within nova.
 *  @param  GammaInitial    The initial values for alpha-vectors (r-n-array).
 *  @param  currentHorizon  The current horizon updated after each iteration.
 *  @param  Gamma           The value of the states (n-array).
 *  @param  GammaPrime      The value of the states (n-array) copy.
 *  @param  pi              The action to take at each state (n-array).
 */
typedef struct NovaPOMDPPBVI {
    float *GammaInitial;

    unsigned int currentHorizon;

    float *Gamma;
    float *GammaPrime;
    unsigned int *pi;
} POMDPPBVI;

/**
 *  Execute all the steps of PBVI for the infinite horizon POMDP model specified.
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVI object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_execute(const POMDP *pomdp, POMDPPBVI *pbvi, POMDPAlphaVectors *policy);

/**
 *  Step 1/4: The initialization step of PBVI. This sets up the Gamma and pi variables.
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVI object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_initialize(const POMDP *pomdp, POMDPPBVI *pbvi);

/**
 *  Step 2/4: The update step of PBVI. This applies the PBVI procedure once.
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVI object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_update(const POMDP *pomdp, POMDPPBVI *pbvi);

/**
 *  Step 3/4: The get resultant policy step of PBVI. This retrieves the alpha-vectors (Gamma) and
 *  corresponding actions (pi).
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVI object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_get_policy(const POMDP *pomdp, POMDPPBVI *pbvi, POMDPAlphaVectors *policy);

/**
 *  Step 4/4: The uninitialization step of PBVI. This sets up the Gamma and pi variables.
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVI object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_uninitialize(const POMDP *pomdp, POMDPPBVI *pbvi);

};

 
#endif // NOVA_POMDP_PBVI_H

