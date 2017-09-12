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


#ifndef NOVA_FUNCTIONS_CPU_H
#define NOVA_FUNCTIONS_CPU_H

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
void ssp_bellman_update_cpu(unsigned int n, unsigned int ns, unsigned int m, 
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
int ssp_random_successor_cpu(const MDP *mdp, unsigned int s, unsigned int a, unsigned int &sp);

/**
 *  Check if the state in the SSP provided is an explicit goal state.
 *  @param  mdp     The MDP object to check.
 *  @param  s       The state to check.
 *  @return True if this is an explicit goal state; False otherwise.
 */
bool ssp_is_goal_cpu(const MDP *mdp, unsigned int s);

/**
 *  Check if the state in the SSP provided is an explicit dead end.
 *  @param  mdp     The MDP object to check.
 *  @param  s       The state to check.
 *  @return True if this is an explicit dead end; False otherwise.
 */
bool ssp_is_dead_end_cpu(const MDP *mdp, unsigned int s);

/**
 *  The SSP stack structure for SSPs.
 *  @param  maxStackSize    The maximum stack size to allocate and check for a stack.
 *  @param  stackSize       The current stack size.
 *  @param  stack           A pointer to the actual stack data in memory.
 */
typedef struct NovaSSPStack {
    unsigned int maxStackSize;
    unsigned int stackSize;
    unsigned int *stack;
} SSPStack;

/**
 *  Create a stack, reserving memory, given the max stack size in the stack object.
 *  @param  stack   The stack object.
 *  @return NOVA_SUCCESS if successful; NOVA_ERROR_INVALID_DATA if the stack was already created.
 */
int ssp_stack_create_cpu(SSPStack &stack);

/**
 *  Pop an element off of the stack.
 *  @param  stack   The stack object.
 *  @param  s       The resultant element when popped off the stack.
 *  @return NOVA_SUCCESS if successful; NOVA_ERROR_EMPTY_CONTAINER if the stack is empty.
 */
int ssp_stack_pop_cpu(SSPStack &stack, unsigned int &s);

/**
 *  Push an element onto the stack.
 *  @param  stack   The stack object.
 *  @param  s       The element to push on the stack.
 *  @return NOVA_SUCCESS if successful; NOVA_ERROR_OUT_OF_MEMORY if the stack is full.
 */
int ssp_stack_push_cpu(SSPStack &stack, unsigned int s);

/**
 *  Check if the element s is in the stack.
 *  @param  stack   The stack object.
 *  @param  s       The element to look for in the stack.
 */
bool ssp_stack_in_cpu(SSPStack &stack, unsigned int s);

/**
 *  Destroy (free) the memory in the stack.
 *  @param  stack   The stack object.
 *  @return NOVA_SUCCESS if successful; NOVA_ERROR_INVALID_DATA if the stack was not yet created.
 */
int ssp_stack_destroy_cpu(SSPStack &stack);

}; // namespace nova

#endif // NOVA_SSP_FUNCTIONS_CPU_H

