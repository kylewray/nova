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


#ifndef MDP_H
#define MDP_H


/*
 *  A structure for an MDP object within nova.
 *  @param  n               The number of states.
 *  @param  ns              The maximum number of successor states.
 *  @param  m               The number of actions.
 *  @param  gamma           The discount factor in [0.0, 1.0).
 *  @param  horizon         The number of iterations to execute (i.e., horizon).
 *  @param  S               A mapping of state-action-successor triples (n-m-ns array) to a
 *                          state index. Reading the array 0 to ns-1, a value of -1 means
 *                          there are no more successors (terminating any loops).
 *  @param  T               A mapping of state-action-successor triples (n-m-ns array) to a
 *                          transition probability.
 *  @param  R               A mapping of state-action pairs (n-m array) to a reward.
 *  @param  currentHorizon  The current horizon updated after each iteration.
 *  @param  V               The value of the states (n-array).
 *  @param  VPrime          The value of the states (n-array) copy.
 *  @param  pi              The action to take at each state (n-array).
 *  @param  d_S             Device-side pointer of S.
 *  @param  d_T             Device-side pointer of T.
 *  @param  d_R             Device-side pointer of R.
 *  @param  d_V             Device-side pointer of V.
 *  @param  d_VPrime        Device-side pointer of VPrime.
 *  @param  d_pi            Device-side pointer of pi.
 */
typedef struct NovaMDP {
    // Core Variables (User-Defined)
    unsigned int n;
    unsigned int ns;
    unsigned int m;

    float gamma;
    unsigned int horizon;

    int *S;
    float *T;
    float *R;

    // Computation Variables (Utilized by Processes Only)
    unsigned int currentHorizon;

    float *V;
    float *VPrime;
    unsigned int *pi;

    int *d_S;
    float *d_T;
    float *d_R;

    float *d_V;
    float *d_VPrime;
    unsigned int *d_pi;
} MDP;


#endif // MDP_H

