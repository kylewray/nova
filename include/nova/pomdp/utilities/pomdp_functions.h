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


#ifndef NOVA_POMDP_FUNCTIONS_H
#define NOVA_POMDP_FUNCTIONS_H


namespace nova {

/**
 *  Compute the dot product of b and alpha.
 *  @param  rz          The maximal number of non-zero elements in B.
 *  @param  Z           The state indexes of the non-zero elements in B.
 *  @param  B           The non-zero beliefs.
 *  @param  bIndex      The particular belief index in B to take the dot product with.
 *  @param  alpha       The alpha vector to take the dot product with.
 *  @return The result of the compuation of b dot alpha.
 */
float pomdp_compute_b_dot_alpha(unsigned int rz, const int *Z, const float *B, unsigned int bIndex, const float *alpha);

/**
 *  Compute the value of V(b) by looking over all alpha-vectors for the max b dot alpha.
 *  @param  n                   The 
 *  @param  rz                  The maximal number of non-zero elements in B.
 *  @param  Z                   The state indexes of the non-zero elements in B.
 *  @param  B                   The non-zero beliefs.
 *  @param  bIndex              The particular belief index in B to take the dot products with for V(b).
 *  @param  Gamma               The set of alpha-vectors of (rGamma-n-array).
 *  @param  rGamma              The number of alpha-vectors in Gamma to consider in computing V(b).
 *  @param  Vnb                 The value of V(b). This will be modified.
 *  @param  alphaPrimeIndex     The index of the maximal alpha-vector in Gamma. This will be modified.
 */
void pomdp_compute_Vb(unsigned int n, unsigned int rz, const int *Z, const float *B, unsigned int bIndex,
    const float *Gamma, unsigned int rGamma, float *Vnb, unsigned int *alphaPrimeIndex);

/**
 *  Perform the Bellman update equation resulting in an alpha-vector and action at a specific belief point.
 *  @param  n           The number of states.
 *  @param  ns          The maximal number of non-zero successors.
 *  @param  m           The number of actions.
 *  @param  z           The number of observations.
 *  @param  r           The number of belief points.
 *  @param  rz          The maximum number of non-zero values among belief points.
 *  @param  gamma       The discount factor in (0, 1).
 *  @param  S           The state transition state indexes of non-zero valued state transitions.
 *  @param  T           The state transition probabilities of non-zero valued state transitions.
 *  @param  O           The observation probabilities.
 *  @param  R           The rewards.
 *  @param  Z           The state indexes of non-zero valued beliefs.
 *  @param  B           The probabilities of non-zero valued beliefs.
 *  @param  Gamma       The set of alpha-vectors.
 *  @param  rGamma      The number of alpha-vectors we consider from Gamma.
 *  @param  bIndex      The particular belief we are considering.
 *  @param  alphaPrime  The resulting alpha-vector. This will be modified. Memory is expected to be allocated.
 *  @param  aPrime      The resulting action for this alpha-vector. This will be modified.
 */
void pomdp_bellman_update(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, float gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    const float *Gamma, unsigned int rGamma,
    unsigned int bIndex, float *alphaPrime, unsigned int *aPrime);

}; // namespace nova


#endif // NOVA_POMDP_FUNCTIONS_H

