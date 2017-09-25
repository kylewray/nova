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


#ifndef NOVA_POMDP_STOCHASTIC_FSC_H
#define NOVA_POMDP_STOCHASTIC_FSC_H


namespace nova {

/*
 *  A structure for POMDP stochastic finite state controller (FSC) policies within nova.
 *  @param  m       The number of actions in the POMDP.
 *  @param  z       The number of observations in the POMDP.
 *  @param  k       The number of controller nodes.
 *  @param  psi     The probabilities each action will be taken in controller nodes (k-m array).
 *  @param  eta     The probabilities of controller node state transitions (k-a-o-k array).
 */
typedef struct NovaPOMDPStochasticFSC {
    unsigned int m;
    unsigned int z;
    unsigned int k;
    float *psi;
    float *eta;
} POMDPStochasticFSC;

/**
 *  Assign variables and allocate the memory *only* for the policy's internal arrays given the parameters.
 *  @param  policy  The stochastic FSC. Arrays within will be created.
 *  @param  k       The number of controller nodes.
 *  @param  m       The number of actions.
 *  @param  z       The number of observations.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_stochastic_fsc_initialize(POMDPStochasticFSC *policy,
    unsigned int k, unsigned int m, unsigned int z);

/**
 *  Randomly sample an action given a controller node.
 *  @param  policy  The stochastic FSC.
 *  @param  q       The current controller node (0 to k-1).
 *  @param  a       The randomly sampled action. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_stochastic_fsc_random_action(POMDPStochasticFSC *policy, unsigned int q, unsigned int &a);

/**
 *  Randomly sample a successor contoller node given a controller node, action taken, and subsequent observation.
 *  @param  policy  The stochastic FSC.
 *  @param  q       The current controller node (0 to k-1).
 *  @param  a       The action taken (0 to m-1).
 *  @param  o       The observation made (0 to z-1).
 *  @param  qp      The randomly sampled successor controller node. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_stochastic_fsc_random_successor(POMDPStochasticFSC *policy,
    unsigned int q, unsigned int a, unsigned int o, unsigned int &qp);

/**
 *  Free the memory for *only* the policy's internal arrays.
 *  @param  policy  The resultant stochastic FSC. Arrays within will be freed.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int pomdp_stochastic_fsc_uninitialize(POMDPStochasticFSC *policy);

};


#endif // NOVA_POMDP_STOCHASTIC_FSC_H


