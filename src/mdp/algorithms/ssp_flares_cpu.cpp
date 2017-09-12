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


#include <nova/mdp/algorithms/ssp_flares_cpu.h>

#include <nova/mdp/policies/mdp_value_function.h>
#include <nova/mdp/utilities/ssp_functions_cpu.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

#include <stdio.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <math.h>
#include <iostream>

namespace nova {

bool ssp_flares_is_solved_cpu(const MDP *mdp, SSPFlaresCPU *flares, unsigned int s)
{
    return std::signbit(flares->V[s]);
}


void ssp_flares_mark_solved_cpu(const MDP *mdp, SSPFlaresCPU *flares, unsigned int s)
{
    flares->V[s] = -std::fabs(flares->V[s]);
}


int ssp_flares_check_solved_cpu(const MDP *mdp, SSPFlaresCPU *flares, unsigned int s0, bool &solved)
{
    solved = true;

    // Create a open state list (stack). Also, create a parallel stack for this state's depth.
    SSPStack open;
    open.maxStackSize = flares->maxStackSize; //mdp->n;
    open.stackSize = 0;
    open.stack = nullptr;

    ssp_stack_create_cpu(open);

    SSPStack openDepth;
    openDepth.maxStackSize = flares->maxStackSize; //mdp->n;
    openDepth.stackSize = 0;
    openDepth.stack = nullptr;

    ssp_stack_create_cpu(openDepth);

    // Create a closed state list (stack). Also, create a parallel stack for this state's depth.
    SSPStack closed;
    closed.maxStackSize = flares->maxStackSize; //mdp->n;
    closed.stackSize = 0;
    closed.stack = nullptr;

    ssp_stack_create_cpu(closed);

    SSPStack closedDepth;
    closedDepth.maxStackSize = flares->maxStackSize; //mdp->n;
    closedDepth.stackSize = 0;
    closedDepth.stack = nullptr;

    ssp_stack_create_cpu(closedDepth);

    // If the initial state is not depth-t-solved, then push it.
    if (!ssp_flares_is_solved_cpu(mdp, flares, s0)) {
        ssp_stack_push_cpu(open, s0);
        ssp_stack_push_cpu(openDepth, 0);
    }

    // Iterate until there are no more elements in the open list.
    while (open.stackSize != 0) {
        // Pop the last element off the open stacks. If we have reached a depth
        // beyond 2t, then we are not solved and continue. Otherwise, keep
        // going and push it on the closed stack.
        unsigned int s = 0;
        unsigned int d = 0;

        ssp_stack_pop_cpu(open, s);
        ssp_stack_pop_cpu(openDepth, d);

        if (d > 2 * flares->t) {
            continue;
        }

        ssp_stack_push_cpu(closed, s);
        ssp_stack_push_cpu(closedDepth, d);

        // Perform a Bellman update on this state if it is not already solved.
        float Vs = flares->V[s];
        if (!ssp_flares_is_solved_cpu(mdp, flares, s)) {
            ssp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->S, mdp->T, mdp->R,
                                   s, flares->V, flares->pi);

            // This lets us compute the residual. We are not solved if it is too large.
            float residual = std::fabs(std::fabs(flares->V[s]) - std::fabs(Vs));
            if (residual >= mdp->epsilon) {
                solved = false;
            }
        }

        // Expand the successors, following the greedy action, but only those that
        // are not solved and are not already in the open or closed sets.
        for (unsigned int i = 0; i < mdp->ns; i++) {
            int sp = mdp->S[s * mdp->m * mdp->ns + flares->pi[s] * mdp->ns + i];
            if (sp < 0) {
                break;
            }

            if (!ssp_flares_is_solved_cpu(mdp, flares, sp)
                    && !ssp_stack_in_cpu(open, sp)
                    && !ssp_stack_in_cpu(closed, sp)) {
                ssp_stack_push_cpu(open, sp);
                ssp_stack_push_cpu(openDepth, d + 1);
            }
        }
    }

    // If this is actually solved, mark all states in the closed set as solved, too.
    // Otherwise, update all of these states.
    while (closed.stackSize != 0) {
        unsigned int s = 0;
        unsigned int d = 0;

        ssp_stack_pop_cpu(closed, s);
        ssp_stack_pop_cpu(closedDepth, d);

        // Note: We require all states to have low residual up to depth 2t. Since successors
        // are only added if the states are not already marked as solved, it will essentially
        // make a tree of max height 2t and label all states less than depth t as solved.
        if (solved) {
            if (d <= flares->t) {
                ssp_flares_mark_solved_cpu(mdp, flares, s);
            }
        } else {
            ssp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->S, mdp->T, mdp->R,
                                   s, flares->V, flares->pi);
        }
    }

    ssp_stack_destroy_cpu(open);
    ssp_stack_destroy_cpu(openDepth);
    ssp_stack_destroy_cpu(closed);
    ssp_stack_destroy_cpu(closedDepth);

    return NOVA_SUCCESS;
}


int ssp_flares_initialize_cpu(const MDP *mdp, SSPFlaresCPU *flares)
{
    if (mdp == nullptr || mdp->n == 0 || flares == nullptr) {
        fprintf(stderr, "Error[ssp_flares_initialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current trial and horizon.
    flares->currentTrial = 0;
    flares->currentHorizon = 0;

    // Create the variables.
    flares->V = new float[mdp->n];
    flares->pi = new unsigned int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    // Note that these values of V are the heuristics for each state.
    // If undefined, then assign 0 for V.
    if (flares->VInitial != nullptr) {
        memcpy(flares->V, flares->VInitial, mdp->n * sizeof(float));
        for (unsigned int i = 0; i < mdp->n; i++) {
            flares->pi[i] = 0;
        }
    } else {
        for (unsigned int i = 0; i < mdp->n; i++) {
            flares->V[i] = 0.0f;
            flares->pi[i] = 0;
        }
    }

    return NOVA_SUCCESS;
}


int ssp_flares_execute_cpu(const MDP *mdp, SSPFlaresCPU *flares, MDPValueFunction *policy)
{
    // First, ensure data is valid.
    if (mdp == nullptr || mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->horizon < 1 || mdp->epsilon < 0.0f ||
            mdp->ng < 1 || mdp->goals == nullptr ||
            flares == nullptr || flares->trials < 1 || policy == nullptr) {
        fprintf(stderr, "Error[ssp_flares_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = ssp_flares_initialize_cpu(mdp, flares);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_flares_execute_cpu]: %s\n", "Failed to initialize the CPU variables.");
        return result;
    }

    // Iterate until you have done the desired number of trials.
    for (flares->currentTrial = 0;
            flares->currentTrial < flares->trials && result != NOVA_CONVERGED;
            flares->currentTrial++) {

        result = ssp_flares_update_cpu(mdp, flares);
        if (result != NOVA_SUCCESS && result != NOVA_CONVERGED) {
            fprintf(stderr, "Error[ssp_flares_execute_cpu]: %s\n",
                            "Failed to perform trial of flares on the CPU.");

            unsigned int resultPrime = ssp_flares_uninitialize_cpu(mdp, flares);
            if (resultPrime != NOVA_SUCCESS) {
                fprintf(stderr, "Error[ssp_flares_execute_cpu]: %s\n",
                                "Failed to uninitialize the CPU variables.");
            }

            return result;
        }
    }

    result = ssp_flares_get_policy_cpu(mdp, flares, policy);
    if (result != NOVA_SUCCESS && result != NOVA_WARNING_APPROXIMATE_SOLUTION) {
        fprintf(stderr, "Error[ssp_flares_execute_cpu]: %s\n", "Failed to get the policy.");

        unsigned int resultPrime = ssp_flares_uninitialize_cpu(mdp, flares);
        if (resultPrime != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_flares_execute_cpu]: %s\n",
                            "Failed to uninitialize the CPU variables.");
        }

        return result;
    }

    bool approximateSolution = (result == NOVA_WARNING_APPROXIMATE_SOLUTION);

    result = ssp_flares_uninitialize_cpu(mdp, flares);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_flares_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
    }

    // If this was an approximate solution, return this warning. Otherwise, return success.
    if (approximateSolution) {
        fprintf(stderr, "Warning[ssp_flares_execute_cpu]: %s\n", "Approximate solution due to early termination and/or dead ends.");
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_SUCCESS;
}


int ssp_flares_uninitialize_cpu(const MDP *mdp, SSPFlaresCPU *flares)
{
    if (mdp == nullptr || flares == nullptr) {
        fprintf(stderr, "Error[ssp_flares_uninitialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon and number of trials.
    flares->currentHorizon = 0;
    flares->currentTrial = 0;

    // Free the memory for V and pi.
    if (flares->V != nullptr) {
        delete [] flares->V;
    }
    flares->V = nullptr;

    if (flares->pi != nullptr) {
        delete [] flares->pi;
    }
    flares->pi = nullptr;

    return NOVA_SUCCESS;
}


int ssp_flares_update_cpu(const MDP *mdp, SSPFlaresCPU *flares)
{
    // Create a visited state list (stack) variable, with just state s0.
    SSPStack visited;
    visited.maxStackSize = mdp->horizon; // flares->maxStackSize;
    visited.stackSize = 0;
    visited.stack = nullptr;

    ssp_stack_create_cpu(visited);

    unsigned int s = mdp->s0;

    for (flares->currentHorizon = 0; flares->currentHorizon < mdp->horizon; flares->currentHorizon++) {
        // Push this new state onto the visited stack. Break if this is absorbing.
        ssp_stack_push_cpu(visited, s);

        // Check if this is a goal or an explicit dead end.
        if (ssp_is_goal_cpu(mdp, s)) {
           flares->V[s] = 0.0f;
           flares->pi[s] = 0;
           break;
        } else if (ssp_is_dead_end_cpu(mdp, s)) {
            flares->V[s] = NOVA_FLT_MAX;
            flares->pi[s] = 0;
           break;
        }

        // Take a greedy action and update the value of this state.
        ssp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m,
                               mdp->S, mdp->T, mdp->R,
                               s, flares->V, flares->pi);

        // Randomly explore the state space using the greedy action.
        unsigned int sp = 0;
        ssp_random_successor_cpu(mdp, s, flares->pi[s], sp);

        // Transition to the next state. Break if this is solved already.
        s = sp;
        if (ssp_flares_is_solved_cpu(mdp, flares, s)) {
            break;
        }
    }

    // At the end, in post order visited, and check for convergence ("solved").
    while (visited.stackSize != 0) {
        // Pop the node off of the traversal stack.
        unsigned int s = 0;
        ssp_stack_pop_cpu(visited, s);

        // Keep checking if we "solved" these states; just return if we find one that is not.
        bool solved = false;
        int result = ssp_flares_check_solved_cpu(mdp, flares, s, solved);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_flares_update_cpu]: %s\n", "Failed to check solved.");
            ssp_stack_destroy_cpu(visited);
            return result;
        }

        if (!solved) {
            ssp_stack_destroy_cpu(visited);
            return NOVA_SUCCESS;
        }
    }

    // If we got here, then we have solved the initial state or run out of time and are done.
    ssp_stack_destroy_cpu(visited);

    if (!ssp_flares_is_solved_cpu(mdp, flares, mdp->s0)) {
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_CONVERGED;
}


int ssp_flares_get_policy_cpu(const MDP *mdp, SSPFlaresCPU *flares, MDPValueFunction *policy)
{
    if (mdp == nullptr || flares == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[ssp_flares_get_policy_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // First, find the best partial solution graph (nodes, which are states).
    // This is stored in the closed state list (below).

    // Create a open state list (stack).
    SSPStack open;
    open.maxStackSize = flares->maxStackSize; //mdp->n;
    open.stackSize = 0;
    open.stack = nullptr;

    ssp_stack_create_cpu(open);
    ssp_stack_push_cpu(open, mdp->s0);

    // Create a closed state list (stack).
    SSPStack closed;
    closed.maxStackSize = flares->maxStackSize; //mdp->n;
    closed.stackSize = 0;
    closed.stack = nullptr;

    ssp_stack_create_cpu(closed);

    // Iterate until there are no more elements in the open list.
    while (open.stackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_stack_pop_cpu(open, s);
        ssp_stack_push_cpu(closed, s);

        // If this is a goal or dead end we do not add successors.
        if (ssp_is_goal_cpu(mdp, s) || ssp_is_dead_end_cpu(mdp, s)) {
            continue;
        }

        // Expand the successors, following the greedy action. Add all successors
        // to the open list.
        for (unsigned int i = 0; i < mdp->ns; i++) {
            int sp = mdp->S[s * mdp->m * mdp->ns + flares->pi[s] * mdp->ns + i];
            if (sp < 0) {
                break;
            }

            // Only add to the stack if it is not already in the open or closed lists.
            if (!ssp_stack_in_cpu(open, sp) && !ssp_stack_in_cpu(closed, sp)) {
                ssp_stack_push_cpu(open, sp);
            }
        }
    }

    // Now we know how many states are in the policy. So, initialize the policy,
    // which allocates memory.
    int result = mdp_value_function_initialize(policy, mdp->n, mdp->m, closed.stackSize);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_flares_get_policy_cpu]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy the final (or intermediate) result, both V and pi.
    for (unsigned int i = 0; i < closed.stackSize; i++) {
        unsigned int s = closed.stack[i];
        policy->S[i] = s;
        policy->V[i] = flares->V[s];
        policy->pi[i] = flares->pi[s];
    }

    ssp_stack_destroy_cpu(open);
    ssp_stack_destroy_cpu(closed);

    // If the initial state is not marked as solved, then return a warning regarding the solution quality.
    if (!ssp_flares_is_solved_cpu(mdp, flares, mdp->s0)) {
        fprintf(stderr, "Warning[ssp_flares_get_policy_cpu]: %s\n",
                "Failed to create a policy without a solved initial state.");
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

