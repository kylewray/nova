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


#ifndef NOVA_SSP_FLARES_H
#define NOVA_SSP_FLARES_H


#include <nova/mdp/mdp.h>
#include <nova/mdp/policies/mdp_value_function.h>

namespace nova {

/**
 *  The necessary variables to perform value iteration on an Flares within nova.
 *  @param  VInitial        The initial value function, mapping states (n-array) to floats.
 *  @param  trials          The number of trials to perform.
 *  @param  t               The depth of the tree to label states as depth-t-solved.
 *  @param  maxStackSize    The maximum stack size for the depth of trials, etc.
 *  @param  currentTrial    The current trial.
 *  @param  currentHorizon  The current horizon updated after each iteration.
 *  @param  V               The value of the states (n-array).
 *  @param  Vprime          The value of the states (n-array) copy.
 *  @param  pi              The action to take at each state (n-array).
 */
typedef struct NovaSSPFlares {
    float *VInitial;
    unsigned int trials;
    unsigned int t;
    unsigned int maxStackSize;

    unsigned int currentTrial;
    unsigned int currentHorizon;

    float *V;
    unsigned int *pi;
} SSPFlares;

/**
 *  Execute all steps of Flares for the SSP MDP model specified.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, Flares also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  flares      The SSPFlares object containing algorithm variables.
 *  @param  policy      The resulting value function policy. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_flares_execute(const MDP *mdp, SSPFlares *flares, MDPValueFunction *policy);

/**
 *  Step 1/4: The initialization step of Flares. This sets up the V and pi variables.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly,
 *  Flares also assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  flares      The SSPFlares object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_flares_initialize(const MDP *mdp, SSPFlares *flares);

/**
 *  Step 2/4: The update step of Flares. This performs one complete trial of Flares.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, Flares also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  flares      The SSPFlares object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_flares_update(const MDP *mdp, SSPFlares *flares);

/**
 *  Step 3/4: The get resultant policy step of Flares. This retrieves the values of states (V) and
 *  the corresponding actions at each state (pi). Unexplored states s will have unchanged
 *  values V(s) and actions pi(s).
 *  Note we assume the rewards R are all positive costs or 0 for goal states.
 *  @param  mdp         The MDP object.
 *  @param  flares      The SSPFlares object containing algorithm variables.
 *  @param  policy      The resulting value function policy. This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_flares_get_policy(const MDP *mdp, SSPFlares *flares, MDPValueFunction *policy);

/**
 *  Step 4/4: The uninitialization step of Flares. This sets up the V and pi variables.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, Flares also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  flares      The SSPFlares object containing algorithm variables.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_flares_uninitialize(const MDP *mdp, SSPFlares *flares);

};


#endif // NOVA_SSP_FLARES_H



