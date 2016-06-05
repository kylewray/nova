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


#ifndef NOVA_SSP_RTDP_CPU_H
#define NOVA_SSP_RTDP_CPU_H


#include <nova/mdp/mdp.h>
#include <nova/mdp/policies/mdp_value_function.h>

namespace nova {

/**
 *  The necessary variables to perform value iteration on an RTDP within nova.
 *  @param  VInitial        The initial value function, mapping states (n-array) to floats.
 *  @param  trials          The number of trials to perform.
 *  @param  currentTrial    The current trial.
 *  @param  currentHorizon  The current horizon updated after each iteration.
 *  @param  V               The value of the states (n-array).
 *  @param  Vprime          The value of the states (n-array) copy.
 *  @param  pi              The action to take at each state (n-array).
 */
typedef struct NovaSSPRTDPCPU {
    float *VInitial;
    unsigned int trials;

    unsigned int currentTrial;
    unsigned int currentHorizon;

    float *V;
    unsigned int *pi;
} SSPRTDPCPU;

/**
 *  Step 1/3: The initialization step of RTDP. This sets up the V and pi variables.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, RTDP also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  rtdp        The SSPRTDPCPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_initialize_cpu(const MDP *mdp, SSPRTDPCPU *rtdp);

/**
 *  Step 2/3: Execute RTDP for the SSP MDP model specified.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, RTDP also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  rtdp        The SSPRTDPCPU object containing algorithm variables.
 *  @param  policy      The resulting value function policy. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_execute_cpu(const MDP *mdp, SSPRTDPCPU *rtdp, MDPValueFunction *policy);

/**
 *  Step 3/3: The uninitialization step of RTDP. This sets up the V and pi variables.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, RTDP also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  rtdp        The SSPRTDPCPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_uninitialize_cpu(const MDP *mdp, SSPRTDPCPU *rtdp);

/**
 *  The update step of RTDP. This performs one complete trial of RTDP.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, RTDP also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  rtdp        The SSPRTDPCPU object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_update_cpu(const MDP *mdp, SSPRTDPCPU *rtdp);

/**
 *  The get resultant policy step of RTDP. This retrieves the values of states (V) and
 *  the corresponding actions at each state (pi). Unexplored states s will have unchanged
 *  values V(s) and actions pi(s).
 *  Note we assume the rewards R are all positive costs or 0 for goal states.
 *  @param  mdp         The MDP object.
 *  @param  rtdp        The SSPRTDPCPU object containing algorithm variables.
 *  @param  policy      The resulting value function policy. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_get_policy_cpu(const MDP *mdp, SSPRTDPCPU *rtdp, MDPValueFunction *policy);

};


#endif // NOVA_SSP_RTDP_CPU_H


