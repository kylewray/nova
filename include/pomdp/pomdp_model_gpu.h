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


/**
 *  Initialize CUDA belief points object.
 *  @param  n       The number of states.
 *  @param  r       The number of belief points.
 *  @param  B       An r-n array, consisting of r sets of n-vector belief distributions.
 *  @param  d_B     An r-n array, consisting of r sets of n-vector belief distributions.
 *                  Device-side pointer.
 *  @return Returns 0 upon success; 1 if invalid arguments were passed; 2 if failed to allocate
            device memory; 3 if failed to copy data.
 */
int pomdp_initialize_belief_points_gpu(unsigned int n, unsigned int r, const float *B,
        float *&d_B);

/**
 *  Uninitialize CUDA belief points object.
 *  @param  d_B     An r-n array, consisting of r sets of n-vector belief distributions.
 *                  Device-side pointer.
 *  @return Returns 0 upon success; 1 if an error occurred with the CUDA functions arose.
 */
int pomdp_uninitialize_belief_points_gpu(float *&d_B);

/**
 *  Initialize CUDA state transitions object.
 *  @param  n       The number of states.
 *  @param  m       The number of actions, in total, that are possible.
 *  @param  T       A mapping of state-action-state triples (n-m-n array) to a
 *                  transition probability.
 *  @param  d_T     A mapping of state-action-state triples (n-m-n array) to a
 *                  transition probability. Device-side pointer.
 *  @return Returns 0 upon success; 1 if invalid arguments were passed; 2 if failed to allocate
            device memory; 3 if failed to copy data.
 */
int pomdp_initialize_state_transitions_gpu(unsigned int n, unsigned int m, const float *T,
        float *&d_T);

/**
 *  Uninitialize CUDA state transitions object.
 *  @param  d_T     A mapping of state-action-state triples (n-m-n array) to a
 *                  transition probability. Device-side pointer.
 *  @return Returns 0 upon success; 1 if an error occurred with the CUDA functions arose.
 */
int pomdp_uninitialize_state_transitions_gpu(float *&d_T);

/**
 *  Initialize CUDA observation transitions object.
 *  @param  n       The number of states.
 *  @param  m       The number of actions, in total, that are possible.
 *  @param  z       The number of observations.
 *  @param  O       A mapping of action-state-observation triples (m-n-z array) to a
 *                  transition probability.
 *  @param  d_O     A mapping of action-state-observation triples (m-n-z array) to a
 *                  transition probability. Device-side pointer.
 *  @return Returns 0 upon success; 1 if invalid arguments were passed; 2 if failed to allocate
            device memory; 3 if failed to copy data.
 */
int pomdp_initialize_observation_transitions_gpu(unsigned int n, unsigned int m,
        unsigned int z, const float *O, float *&d_O);

/**
 *  Uninitialize CUDA observation transitions object.
 *  @param  d_O     A mapping of action-state-observation triples (m-n-z array) to a
 *                  transition probability. Device-side pointer.
 *  @return Returns 0 upon success; 1 if an error occurred with the CUDA functions arose.
 */
int pomdp_uninitialize_observation_transitions_gpu(float *&d_O);

/**
 *  Initialize CUDA rewards object.
 *  @param  n       The number of states.
 *  @param  m       The number of actions, in total, that are possible.
 *  @param  R       A mapping of state-action pairs (n-m array) to a reward.
 *  @param  d_R     A mapping of state-action pairs (n-m array) to a reward.
 *                  Device-side pointer.
 *  @return Returns 0 upon success; 1 if invalid arguments were passed; 2 if failed to allocate
            device memory; 3 if failed to copy data.
 */
int pomdp_initialize_rewards_gpu(unsigned int n, unsigned int m, const float *R, float *&d_R);

/**
 *  Uninitialize CUDA rewards object.
 *  @param  d_R     A mapping of state-action pairs (n-m array) to a reward. Device-side pointer.
 *  @return Returns 0 upon success; 1 if an error occurred with the CUDA functions arose.
 */
int pomdp_uninitialize_rewards_gpu(float *&d_R);

/**
 *  Initialize CUDA available actions object.
 *  @param  m               The number of actions, in total, that are possible.
 *  @param  r               The number of belief points.
 *  @param  available       A r-m array, consisting of r sets of m-vector availability values.
 *  @param  d_available     A r-m array, consisting of r sets of m-vector availability values.
 *                          Device-side pointer.
 *  @return Returns 0 upon success; 1 if invalid arguments were passed; 2 if failed to allocate
            device memory; 3 if failed to copy data.
 */
int pomdp_initialize_available_gpu(unsigned int m, unsigned int r, const bool *available,
        bool *&d_available);

/**
 *  Uninitialize CUDA available actions object.
 *  @param  d_available     A r-m array, consisting of r sets of m-vector availability values.
 *                          Device-side pointer.
 *  @return Returns 0 upon success; 1 if an error occurred with the CUDA functions arose.
 */
int pomdp_uninitialize_available_gpu(bool *&d_available);

/**
 *  Initialize CUDA non-zero belief states object.
 *  @param  r                  The number of belief points.
 *  @param  maxNonZeroBeliefs  The maximum number of non-zero belief states.
 *  @param  nonZeroBeliefs     A mapping of beliefs to an array of state indexes;
 *                             -1 denotes the end of the array.
 *  @param  d_nonZeroBeliefs   A mapping of beliefs to an array of state indexes;
 *                             -1 denotes the end of the array. Device-side pointer.
 *  @return Returns 0 upon success; 1 if invalid arguments were passed; 2 if failed to allocate
            device memory; 3 if failed to copy data.
 */
int pomdp_initialize_nonzero_beliefs_gpu(unsigned int r, unsigned int maxNonZeroBeliefs,
        const int *nonZeroBeliefs, int *&d_nonZeroBeliefs);

/**
 *  Uninitialize CUDA non-zero belief states object.
 *  @param  d_NonZeroBeliefs    A mapping of beliefs to an array of state indexes;
 *                              -1 denotes the end of the array. Device-side pointer.
 *  @return Returns 0 upon success; 1 if an error occurred with the CUDA functions arose.
 */
int pomdp_uninitialize_nonzero_beliefs_gpu(int *&d_nonZeroBeliefs);

/**
 *  Initialize CUDA by transferring all of the constant POMDP model information to the device.
 *  @param  n               The number of states.
 *  @param  m               The number of actions, in total, that are possible.
 *  @param  maxSuccessors   The maximum number of successor states.
 *  @param  successors      A mapping of state-action pairs a set of possible successor states;
 *                          -1 denotes the end of the array.
 *  @param  d_successors    A mapping of state-action pairs a set of possible successor states;
 *                          -1 denotes the end of the array. Device-side pointer.
 *  @return Returns 0 upon success; 1 if invalid arguments were passed; 2 if failed to allocate
            device memory; 3 if failed to copy data.
 */
int pomdp_initialize_successors_gpu(unsigned int n, unsigned int m,
        unsigned int maxSuccessors, const int *successors, int *&d_successors);

/**
 *  Uninitialize CUDA by transferring all of the constant POMDP model information to the device.
 *  @param  d_successors    A mapping of state-action pairs a set of possible successor states;
 *                          -1 denotes the end of the array. Device-side pointer.
 *  @return Returns 0 upon success; 1 if an error occurred with the CUDA functions arose.
 */
int pomdp_uninitialize_successors_gpu(int *&d_successors);


#endif // NOVA_POMDP_MODEL_GPU_H

