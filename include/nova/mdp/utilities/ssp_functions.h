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


#ifndef NOVA_SSP_FUNCTIONS_H
#define NOVA_SSP_FUNCTIONS_H

#include <nova/mdp/mdp.h>

namespace nova {

/**
 *  Apply a Bellman update at the state provided and update the value and policy.
 *  @param  n   The number of states.
 *  @param  ns  The maximum number of successor states.
 *  @param  m   The number of actions.
 *  @param  S   The array of indexes with non-zero state transition probabilities.
 *  @param  T   The array of probabilities with non-zero values.
 *  @param  R   The reward function.
 *  @param  s   The state on which to perform the Bellman update.
 *  @param  V   The value of the states. This will be modified at the state provided.
 *  @param  pi  The policy. This will be modified at the state provided.
 */
void ssp_bellman_update(unsigned int n, unsigned int ns, unsigned int m, 
    const int *S, const float *T, const float *R, unsigned int s,
    float *V, unsigned int *pi);

/**
 *  Generate a random successor from the SSP provided given a state-action pair.
 *  @param  mdp     The MDP object.
 *  @param  s       The current state.
 *  @param  a       The action taken.
 *  @param  sp      The successor state.
 *  @return NOVA_SUCCESS if successful; NOVA_ERROR_INVALID_DATA if the inputs are invalid.
 */
int ssp_random_successor(const MDP *mdp, unsigned int s, unsigned int a, unsigned int &sp);

/**
 *  Check if the state in the SSP provided is an explicit goal state.
 *  @param  mdp     The MDP object to check.
 *  @param  s       The state to check.
 *  @return True if this is an explicit goal state; False otherwise.
 */
bool ssp_is_goal(const MDP *mdp, unsigned int s);

/**
 *  Check if the state in the SSP provided is an explicit dead end.
 *  @param  mdp     The MDP object to check.
 *  @param  s       The state to check.
 *  @return True if this is an explicit dead end; False otherwise.
 */
bool ssp_is_dead_end(const MDP *mdp, unsigned int s);

}; // namespace nova

#endif // NOVA_SSP_FUNCTIONS_H

