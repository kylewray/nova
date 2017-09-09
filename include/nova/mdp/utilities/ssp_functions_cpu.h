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

typedef struct NovaSSPStack {
    unsigned int maxStackSize;
    unsigned int stackSize;
    unsigned int *stack;
} SSPStack;

void ssp_bellman_update_cpu(unsigned int n, unsigned int ns, unsigned int m, 
    const int *S, const float *T, const float *R, unsigned int s,
    float *V, unsigned int *pi);
void ssp_random_successor_cpu(const MDP *mdp, unsigned int s, unsigned int a, unsigned int &sp);
bool ssp_is_goal_cpu(const MDP *mdp, unsigned int s);
bool ssp_is_dead_end_cpu(const MDP *mdp, unsigned int s);

void ssp_stack_create_cpu(SSPStack &stack);
void ssp_stack_pop_cpu(SSPStack &stack, unsigned int &s);
void ssp_stack_push_cpu(SSPStack &stack, unsigned int s);
bool ssp_stack_in_cpu(SSPStack &stack, unsigned int s);
void ssp_stack_destroy_cpu(SSPStack &stack);

}; // namespace nova

#endif // NOVA_SSP_FUNCTIONS_CPU_H

