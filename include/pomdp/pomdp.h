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


#ifndef POMDP_H
#define POMDP_H


/*
 *  A structure for a POMDP object within nova.
 *
 *  Only define the variables n through B. All others are nullptr by default, and are
 *  cleaned up by calling the relevant uninitialize functions. The generic uninitialize_pomdp
 *  function will free everything, instead of having to manage all other variables yourself.
 *
 *  @param  n               The number of states.
 *  @param  ns              The maximum number of successor states possible.
 *  @param  m               The number of actions, in total, that are possible.
 *  @param  z               The number of observations.
 *  @param  r               The number of belief points.
 *  @param  rz              The maximum number of non-zero belief states.
 *  @param  gamma           The discount factor in [0.0, 1.0).
 *  @param  horizon         The number of iterations to execute (i.e., horizon).
 *  @param  S               A mapping of state-action-successor triples (n-m-ns array) to a state
 *                          index. Reading the array 0 to ns-1, a value of -1 means there are
 *                          no more successors (terminating any loops).
 *  @param  T               A mapping of state-action-successor triples (n-m-ns array) to a
 *                          transition probability.
 *  @param  O               A mapping of action-state-observations triples (m-n-z array) to a
 *                          transition probability.
 *  @param  R               A mapping of state-action triples (n-m array) to a reward.
 *  @param  Z               A mapping of belief-non-zero-state pairs (r-rz array) to a state index.
 *                          Reading the array 0 to rz-1, a value of -1 means there are no more
 *                          states with a non-zero belief value (terminating any loops).
 *  @param  B               A mapping of belief-non-zero-state pairs (r-rz array) consisting of r
 *                          sets of rz-vector belief distributions.
 *  @param  Gamma           The resultant policy's alpha-vectors (r-n array). CPU version only.
 *  @param  GammaPrime      The resultant policy's alpha-vectors (r-n array). CPU version only.
 *  @param  pi              The resultant policy's actions (r array). CPU version only.
 *  @param  piPrime         The resultant policy's actions (r array). CPU version only.
 *  @param  d_S             Device-side pointer of S. GPU version only.
 *  @param  d_T             Device-side pointer of T. GPU version only.
 *  @param  d_O             Device-side pointer of O. GPU version only.
 *  @param  d_R             Device-side pointer of R. GPU version only.
 *  @param  d_Z             Device-side pointer of Z. GPU version only.
 *  @param  d_B             Device-side pointer of B. GPU version only.
 *  @param  d_Gamma         Device-side pointer of Gamma. GPU version only.
 *  @param  d_GammaPrime    Device-side pointer of Gamma. GPU version only.
 *  @param  d_pi            Device-side pointer of pi. GPU version only.
 *  @param  d_piPrime       Device-side pointer of pi. GPU version only.
 *  @param  d_alphaBA       Device-side pointer; intermediate alpha-vectors. GPU version only.
 */
typedef struct NovaPOMDP {
    // Core Variables (User-Defined)
    unsigned int n;
    unsigned int ns;
    unsigned int m;
    unsigned int z;
    unsigned int r;
    unsigned int rz;

    float gamma;
    unsigned int horizon;

    int *S;
    float *T;
    float *O;
    float *R;
    int *Z;
    float *B;

    // Computation Variables (Utilized by Processes Only)
    float *Gamma;
    float *GammaPrime;
    unsigned int *pi;
    unsigned int *piPrime;

    int *d_S;
    float *d_T;
    float *d_O;
    float *d_R;
    int *d_Z;
    float *d_B;

    float *d_Gamma;
    float *d_GammaPrime;
    unsigned int *d_pi;
    unsigned int *d_piPrime;
    float *d_alphaBA;
} POMDP;


#endif // POMDP_H


