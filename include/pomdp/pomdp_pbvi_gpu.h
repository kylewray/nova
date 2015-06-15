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


#ifndef POMDP_PBVI_GPU_H
#define POMDP_PBVI_GPU_H


/**
 *  Execute the entire PBVI process for the infinite horizon POMDP model specified using the GPU.
 *  @param  n                   The number of states.
 *  @param  ns                  The maximum number of successor states possible.
 *  @param  m                   The number of actions, in total, that are possible.
 *  @param  z                   The number of observations.
 *  @param  r                   The number of belief points.
 *  @param  rz                  The maximum number of non-zero belief states.
 *  @param  Z                   A mapping of belief-non-zero-state pairs (r-rz array) to a state index.
 *                              Reading the array 0 to rz-1, a value of -1 means there are no more
 *                              states with a non-zero belief value (terminating any loops).
 *  @param  B                   A mapping of belief-non-zero-state pairs (r-rz array) consisting of r
 *                              sets of rz-vector belief distributions.
 *  @param  S                   A mapping of state-action-successor triples (n-m-ns array) to a state
 *                              index. Reading the array 0 to ns-1, a value of -1 means there are
 *                              no more successors (terminating any loops).
 *  @param  T                   A mapping of state-action-successor triples (n-m-ns array) to a
 *                              transition probability.
 *  @param  O                   A mapping of action-state-observations triples (m-n-z array) to a
 *                              transition probability.
 *  @param  R                   A mapping of state-action triples (n-m array) to a reward.
 *  @param  gamma               The discount factor in [0.0, 1.0).
 *  @param  horizon             How many time steps to iterate.
 *  @param  numThreads          The number of CUDA threads per block. Use multiples of 32.
 *  @param  Gamma               The resultant policy; set of alpha vectors (r-n array).
                                This will be modified.
 *  @param  pi                  The resultant policy; one action for each alpha-vector (r-array).
 *                              This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_pbvi_complete_gpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z, unsigned int r, unsigned int nz,
        const int *Z, const float *B, const int *S, const float *T, const float *O, const float *R,
        float gamma, unsigned int horizon, unsigned int numThreads,
        float *Gamma, unsigned int *pi);

/**
 *  Step 1/3: The initialization step of PBVI. This sets up the Gamma, pi, alphaBA, and numBlocks variables.
 *  @param  n               The number of states.
 *  @param  m               The number of actions, in total, that are possible.
 *  @param  r               The number of belief points.
 *  @param  numThreads      The number of CUDA threads per block. Use multiples of 32.
 *  @param  Gamma           The resultant policy; set of alpha vectors (r-n array).
 *  @param  d_Gamma         An r-n array of the alpha-vectors.
                            Device-side pointer. This will be modified.
 *  @param  d_GammaPrime    An r-n array of the alpha-vectors (copied).
                            Device-side pointer. This will be modified.
 *  @param  d_pi            An r-array of the actions at each belief.
                            Device-side pointer. This will be modified.
 *  @param  d_piPrime       An r-array of the actions at each belief (copied).
                            Device-side pointer. This will be modified.
 *  @param  d_alphaBA       A set of intermediate alpha-vectors.
                            Device-side pointer. This will be modified.
 *  @param  numBlocks       The number of blocks to execute. This will be modified.
 *  @return Returns 0 upon success, 1 otherwise.
 */
int pomdp_pbvi_initialize_gpu(unsigned int n, unsigned int m, unsigned int r,
        unsigned int numThreads, float *Gamma, float *&d_Gamma, float *&d_GammaPrime,
        unsigned int *&d_pi, unsigned int *&d_piPrime, float *&d_alphaBA,
        unsigned int *numBlocks);

/**
 *  The update step of PBVI. This applies the PBVI procedure once.
 *  @param  n                   The number of states.
 *  @param  ns                  The maximum number of successor states possible.
 *  @param  m                   The number of actions, in total, that are possible.
 *  @param  z                   The number of observations.
 *  @param  r                   The number of belief points.
 *  @param  rz                  The maximum number of non-zero belief states.
 *  @param  d_Z                 A mapping of belief-non-zero-state pairs (r-rz array) to a state index.
 *                              Reading the array 0 to rz-1, a value of -1 means there are no more
 *                              states with a non-zero belief value (terminating any loops).
 *                              Device-side pointer.
 *  @param  d_B                 A mapping of belief-non-zero-state pairs (r-rz array) consisting of r
 *                              sets of rz-vector belief distributions. Device-side pointer.
 *  @param  d_S                 A mapping of state-action-successor triples (n-m-ns array) to a state
 *                              index. Reading the array 0 to ns-1, a value of -1 means there are
 *                              no more successors (terminating any loops). Device-side pointer.
 *  @param  d_T                 A mapping of state-action-successor triples (n-m-ns array) to a
 *                              transition probability. Device-side pointer.
 *  @param  d_O                 A mapping of action-state-observations triples (m-n-z array) to a
 *                              transition probability. Device-side pointer.
 *  @param  d_R                 A mapping of state-action triples (n-m array) to a reward.
 *                              Device-side pointer.
 *  @param  gamma               The discount factor in [0.0, 1.0).
 *  @param  currentHorizon      How many applications of this method have been applied so far.
 *  @param  numThreads          The number of CUDA threads per block. Use multiples of 32.
 *  @param  numBlocks           The number of blocks to execute.
 *  @param  d_Gamma             An r-n array of the alpha-vectors. Device-side pointer.
 *  @param  d_GammaPrime        An r-n array of the alpha-vectors (copied). Device-side pointer.
 *  @param  d_pi                An r-array of the actions at each belief. Device-side pointer.
 *  @param  d_piPrime           An r-array of the actions at each belief (copied).
                                Device-side pointer.
 *  @param  d_alphaBA           A set of intermediate alpha-vectors. Device-side pointer.
 *  @return Returns 0 upon success, 1 otherwise.
 */
int pomdp_pbvi_update_gpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z, unsigned int r, unsigned int rz,
        const int *d_Z, const float *d_B, const int *d_S, const float *d_T, const float *d_O, const float *d_R,
        float gamma, unsigned int currentHorizon, unsigned int numThreads, unsigned int numBlocks,
        float *d_Gamma, float *d_GammaPrime, unsigned int *d_pi, unsigned int *d_piPrime,
        float *d_alphaBA);

/**
 *  The get resultant policy step of PBVI. This retrieves the alpha-vectors (Gamma) and
 *  corresponding actions (pi).
 *  @param n            The number of states.
 *  @param r            The number of belief points.
 *  @param horizon      How many time steps to iterate.
 *  @param d_Gamma      An r-n array of the alpha-vectors. Device-side pointer.
 *  @param d_GammaPrime An r-n array of the alpha-vectors (copied). Device-side pointer.
 *  @param d_pi         An r-array of the actions at each belief. Device-side pointer.
 *  @param d_piPrime    An r-array of the actions at each belief (copied). Device-side pointer.
 *  @param d_alphaBA    A set of intermediate alpha-vectors. Device-side pointer.
 *  @param Gamma        The resultant policy; set of alpha vectors (r-n array).
                        This will be modified.
 *  @param pi           The resultant policy; one action for each alpha-vector (r-array).
                        This will be modified.
 *  @return Returns 0 upon success, 1 otherwise.
 */
int pomdp_pbvi_get_policy_gpu(unsigned int n, unsigned int r, unsigned int horizon,
        const float *d_Gamma, const float *d_GammaPrime, const unsigned int *d_pi,
        const unsigned int *d_piPrime, float *Gamma, unsigned int *pi);

/**
 *  Step 3/3: The uninitialization step of PBVI. This sets up the Gamma, pi, alphaBA, and numBlocks variables.
 *  @param  d_Gamma         An r-n array of the alpha-vectors.
                            Device-side pointer. This will be modified.
 *  @param  d_GammaPrime    An r-n array of the alpha-vectors (copied).
                            Device-side pointer. This will be modified.
 *  @param  d_pi            An r-array of the actions at each belief.
                            Device-side pointer. This will be modified.
 *  @param  d_piPrime       An r-array of the actions at each belief (copied).
                            Device-side pointer. This will be modified.
 *  @param  d_alphaBA       A set of intermediate alpha-vectors.
                            Device-side pointer. This will be modified.
 *  @return Returns 0 upon success, 1 otherwise.
 */
int pomdp_pbvi_uninitialize_gpu(float *&d_Gamma, float *&d_GammaPrime,
        unsigned int *&d_pi, unsigned int *&d_piPrime, float *&d_alphaBA);

/**
 *  Step 2/3: Execute PBVI for the infinite horizon POMDP model specified.
 *  @param  n               The number of states.
 *  @param  ns              The maximum number of successor states possible.
 *  @param  m               The number of actions, in total, that are possible.
 *  @param  z               The number of observations.
 *  @param  r               The number of belief points.
 *  @param  rz              The maximum number of non-zero belief states.
 *  @param  d_Z             A mapping of belief-non-zero-state pairs (r-rz array) to a state index.
 *                          Reading the array 0 to rz-1, a value of -1 means there are no more
 *                          states with a non-zero belief value (terminating any loops).
 *                          Device-side pointer.
 *  @param  d_B             A mapping of belief-non-zero-state pairs (r-rz array) consisting of r
 *                          sets of rz-vector belief distributions. Device-side pointer.
 *  @param  d_S             A mapping of state-action-successor triples (n-m-ns array) to a state
 *                          index. Reading the array 0 to ns-1, a value of -1 means there are
 *                          no more successors (terminating any loops). Device-side pointer.
 *  @param  d_T             A mapping of state-action-successor triples (n-m-ns array) to a
 *                          transition probability. Device-side pointer.
 *  @param  d_O             A mapping of action-state-observations triples (m-n-z array) to a
 *                          transition probability. Device-side pointer.
 *  @param  d_R             A mapping of state-action triples (n-m array) to a reward.
 *                          Device-side pointer.
 *  @param  gamma           The discount factor in [0.0, 1.0).
 *  @param  horizon         How many time steps to iterate.
 *  @param  numThreads      The number of CUDA threads per block. Use multiples of 32.
 *  @param  Gamma           The resultant policy; set of alpha vectors (r-n array).
                            This will be modified.
 *  @param  pi              The resultant policy; one action for each alpha-vector (r-array).
 *                          This will be modified.
 *  @return Returns 0 upon success, 1 otherwise.
 */
int pomdp_pbvi_execute_gpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z, unsigned int r, unsigned int rz,
        const int *d_Z, const float *d_B, const int *d_S, const float *d_T, const float *d_O, const float *d_R,
        float gamma, unsigned int horizon, unsigned int numThreads,
        float *Gamma, unsigned int *pi);


#endif // POMDP_PBVI_GPU_H

