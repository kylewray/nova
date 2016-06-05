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


#ifndef NOVA_POMDP_PBVI_GPU_H
#define NOVA_POMDP_PBVI_GPU_H


#include <nova/pomdp/pomdp.h>
#include <nova/pomdp/policies/pomdp_alpha_vectors.h>

namespace nova {

/**
 *  The necessary variables to perform point-based value iteration on a POMDP within nova.
 *  @param  GammaInitial    The initial values for alpha-vectors (r-n-array).
 *  @param  numThreads      The number of CUDA threads per block. Use multiples of 32.
 *  @param  currentHorizon  The current horizon updated after each iteration.
 *  @param  d_Gamma         The value of the states (n-array), a device-side pointer.
 *  @param  d_GammaPrime    The value of the states (n-array) copy, a device-side pointer.
 *  @param  d_pi            The action to take at each state (n-array), a device-side pointer.
 *  @param  d_alphaBA       The intermediate alpha-vectors, a device-side pointer.
 */
typedef struct NovaPOMDPPBVIGPU {
    float *GammaInitial;
    unsigned int numThreads;

    unsigned int currentHorizon;

    float *d_Gamma;
    float *d_GammaPrime;
    unsigned int *d_pi;

    float *d_alphaBA;
} POMDPPBVIGPU;

/**
 *  Step 1/3: The initialization step of PBVI. This sets up the Gamma, pi, and alphaBA variables.
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVIGPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_initialize_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi);

/**
 *  Step 2/3: Execute PBVI for the infinite horizon POMDP model specified.
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVIGPU object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be created and modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_execute_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi,
        POMDPAlphaVectors *&policy);

/**
 *  Step 3/3: The uninitialization step of PBVI. This sets up the Gamma, pi, and alphaBA variables.
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVIGPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_uninitialize_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi);

/**
 *  The update step of PBVI. This applies the PBVI procedure once.
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVIGPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_update_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi);

/**
 *  The get resultant policy step of PBVI. This retrieves the alpha-vectors (Gamma) and
 *  corresponding actions (pi).
 *  @param  pomdp       The POMDP object.
 *  @param  pbvi        The POMDPPBVIGPU object containing algorithm variables.
 *  @param  policy      The resultant set of alpha-vectors. This will be created and modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_get_policy_gpu(const POMDP *pomdp, POMDPPBVIGPU *pbvi,
        POMDPAlphaVectors *&policy);

};


#endif // NOVA_POMDP_PBVI_GPU_H

