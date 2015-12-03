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


#ifndef SSP_RTDP_CPU_H
#define SSP_RTDP_CPU_H


#include "mdp.h"


/**
 *  Execute the entire RTDP process for the SSP MDP model specified until convergence. Uses the CPU.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, RTDP also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp     The MDP object.
 *  @param  Vinitial    The initial value function, mapping states (n-array) to floats. Importantly,
 *                      this should initially hold the *admissible heuristic* values for each state.
 *  @param  r           The number of states relevant to the final policy with r <= n.
 *  @param  S           The set of relevant state indexes (r-array). This will be modified.
 *  @param  V           The resultant policy; one value for each state (r-array). This will be modified.
 *  @param  pi          The resultant policy; one action for each state (r-array). This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_complete_cpu(MDP *mdp, const float *Vinitial, unsigned int &r,
    unsigned int *&S, float *&V, unsigned int *&pi);

/**
 *  Step 1/3: The initialization step of RTDP. This sets up the V and pi variables.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, RTDP also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  Vinitial    The initial value function, mapping states (n-array) to floats. Importantly,
 *                      this should initially hold the *admissible heuristic* values for each state.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_initialize_cpu(MDP *mdp, const float *Vinitial);

/**
 *  Step 2/3: Execute RTDP for the SSP MDP model specified.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, RTDP also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp         The MDP object.
 *  @param  Vinitial    The initial value function, mapping states (n-array) to floats. Importantly,
 *                      this should initially hold the *admissible heuristic* values for each state.
 *  @param  r           The number of states relevant to the final policy with r <= n.
 *  @param  S           The set of relevant state indexes (r-array). This will be modified.
 *  @param  V           The resultant policy; one value for each state (r-array). This will be modified.
 *  @param  pi          The resultant policy; one action for each state (r-array). This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_execute_cpu(MDP *mdp, const float *Vinitial, unsigned int &r,
    unsigned int *&S, float *&V, unsigned int *&pi);

/**
 *  Step 3/3: The uninitialization step of RTDP. This sets up the V and pi variables.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, RTDP also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp     The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_uninitialize_cpu(MDP *mdp);

/**
 *  The update step of RTDP. This performs one trial of RTDP.
 *  Note we assume the rewards R are all positive costs or 0 for goal states. Importantly, RTDP also
 *  assumes that the goal can be reached with non-zero probability from all states.
 *  @param  mdp     The MDP object.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_update_cpu(MDP *mdp);

/**
 *  The get resultant policy step of RTDP. This retrieves the values of states (V) and
 *  the corresponding actions at each state (pi). Unexplored states s will have unchanged
 *  values V(s) and actions pi(s).
 *  Note we assume the rewards R are all positive costs or 0 for goal states.
 *  @param  mdp     The MDP object.
 *  @param  r       The number of states relevant to the final policy with r <= n.
 *  @param  S       The set of relevant state indexes (r-array). This will be modified.
 *  @param  V       The resultant policy; one value for each relevant state (r-array). This will be modified.
 *  @param  pi      The resultant policy; one action for each relevant state (r-array). This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_get_policy_cpu(MDP *mdp, unsigned int &r, unsigned int *&S,
    float *&V, unsigned int *&pi);

/**
 *  Free the policy produced by the get policy above.
 *  @param  S       The set of relevant state indexes (r-array). This will be modified.
 *  @param  V       The resultant policy; one value for each relevant state (r-array). This will be modified.
 *  @param  pi      The resultant policy; one action for each relevant state (r-array). This will be modified.
 *  @return Returns zero upon success, non-zero otherwise.
 */
extern "C" int ssp_rtdp_free_policy_cpu(MDP *mdp, unsigned int *&S, float *&V, unsigned int *&pi);


#endif // SSP_RTDP_CPU_H


