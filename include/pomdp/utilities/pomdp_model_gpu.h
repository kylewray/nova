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


#ifndef NOVA_POMDP_MODEL_GPU_H
#define NOVA_POMDP_MODEL_GPU_H


#include "pomdp.h"

namespace nova {

/**
 *  Initialize CUDA by transferring all of the constant POMDP model information to the device.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_initialize_gpu(POMDP *pomdp);

/**
 *  Uninitialize CUDA by transferring all of the constant POMDP model information to the device.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_uninitialize_gpu(POMDP *pomdp);

/**
 *  Initialize CUDA successors object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_initialize_successors_gpu(POMDP *pomdp);

/**
 *  Uninitialize CUDA successors object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_uninitialize_successors_gpu(POMDP *pomdp);

/**
 *  Initialize CUDA state transitions object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_initialize_state_transitions_gpu(POMDP *pomdp);

/**
 *  Uninitialize CUDA state transitions object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_uninitialize_state_transitions_gpu(POMDP *pomdp);

/**
 *  Initialize CUDA observation transitions object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_initialize_observation_transitions_gpu(POMDP *pomdp);

/**
 *  Uninitialize CUDA observation transitions object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_uninitialize_observation_transitions_gpu(POMDP *pomdp);

/**
 *  Initialize CUDA rewards object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_initialize_rewards_gpu(POMDP *pomdp);

/**
 *  Uninitialize CUDA rewards object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_uninitialize_rewards_gpu(POMDP *pomdp);

/**
 *  Initialize CUDA non-zero belief states object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_initialize_nonzero_beliefs_gpu(POMDP *pomdp);

/**
 *  Uninitialize CUDA non-zero belief states object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_uninitialize_nonzero_beliefs_gpu(POMDP *pomdp);

/**
 *  Initialize CUDA belief points object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_initialize_belief_points_gpu(POMDP *pomdp);

/**
 *  Uninitialize CUDA belief points object.
 *  @param  pomdp   The POMDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_uninitialize_belief_points_gpu(POMDP *pomdp);

};


#endif // NOVA_POMDP_MODEL_GPU_H

