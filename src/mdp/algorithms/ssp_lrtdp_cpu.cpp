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


#include <nova/mdp/algorithms/ssp_lrtdp_cpu.h>

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

bool ssp_lrtdp_is_solved_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, unsigned int s)
{
    return std::signbit(lrtdp->V[s]);
}


void ssp_lrtdp_mark_solved_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, unsigned int s)
{
    lrtdp->V[s] = -std::fabs(lrtdp->V[s]);
}


int ssp_lrtdp_check_depth_limited_solved_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, unsigned int s0, bool &solved)
{
    solved = true;

    // Create a open state list (stack).
    SSPStack open;
    open.maxStackSize = mdp->n;
    open.stackSize = 0;
    open.stack = nullptr;

    ssp_stack_create_cpu(open);

    // Create a closed state list (stack).
    SSPStack closed;
    closed.maxStackSize = mdp->n;
    closed.stackSize = 0;
    closed.stack = nullptr;

    ssp_stack_create_cpu(closed);

    // If the initial state is not solved, then push it.
    if (!ssp_lrtdp_is_solved_cpu(mdp, lrtdp, s0)) {
        ssp_stack_push_cpu(open, s0);
    }

    // Iterate until there are no more elements in the open list.
    while (open.stackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_stack_pop_cpu(open, s);
        ssp_stack_push_cpu(closed, s);

        // Perform a Bellman update on this state if it is not already solved.
        float Vs = lrtdp->V[s];
        if (!ssp_lrtdp_is_solved_cpu(mdp, lrtdp, s)) {
            ssp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->S, mdp->T, mdp->R,
                                   s, lrtdp->V, lrtdp->pi);

            // This lets us compute the residual. We are not solved if it is too large.
            float residual = std::fabs(std::fabs(lrtdp->V[s]) - std::fabs(Vs));
            if (residual >= mdp->epsilon) {
                solved = false;
            }
        }

        // Expand the successors, following the greedy action, but only those that
        // are not solved and are not already in the open or closed sets.
        for (unsigned int i = 0; i < mdp->ns; i++) {
            int sp = mdp->S[s * mdp->m * mdp->ns + lrtdp->pi[s] * mdp->ns + i];
            if (sp < 0) {
                break;
            }

            if (!ssp_lrtdp_is_solved_cpu(mdp, lrtdp, sp)
                    && !ssp_stack_in_cpu(open, sp)
                    && !ssp_stack_in_cpu(closed, sp)) {
                ssp_stack_push_cpu(open, sp);
            }
        }
    }

    // If this is actually solved, mark all states in the closed set as solved, too.
    // Otherwise, update all of these states.
    while (closed.stackSize != 0) {
        unsigned int s = 0;
        ssp_stack_pop_cpu(closed, s);
        if (solved) {
            ssp_lrtdp_mark_solved_cpu(mdp, lrtdp, s);
        } else {
            ssp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->S, mdp->T, mdp->R,
                                   s, lrtdp->V, lrtdp->pi);
        }
    }

    ssp_stack_destroy_cpu(open);
    ssp_stack_destroy_cpu(closed);

    return NOVA_SUCCESS;
}


int ssp_lrtdp_initialize_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp)
{
    if (mdp == nullptr || mdp->n == 0 || lrtdp == nullptr) {
        fprintf(stderr, "Error[ssp_lrtdp_initialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current trial and horizon.
    lrtdp->currentTrial = 0;
    lrtdp->currentHorizon = 0;

    // Create the variables.
    lrtdp->V = new float[mdp->n];
    lrtdp->pi = new unsigned int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    // Note that these values of V are the heuristics for each state.
    // If undefined, then assign 0 for V.
    if (lrtdp->VInitial != nullptr) {
        memcpy(lrtdp->V, lrtdp->VInitial, mdp->n * sizeof(float));
        for (unsigned int i = 0; i < mdp->n; i++) {
            lrtdp->pi[i] = 0;
        }
    } else {
        for (unsigned int i = 0; i < mdp->n; i++) {
            lrtdp->V[i] = 0.0f;
            lrtdp->pi[i] = 0;
        }
    }

    return NOVA_SUCCESS;
}


int ssp_lrtdp_execute_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, MDPValueFunction *policy)
{
    // First, ensure data is valid.
    if (mdp == nullptr || mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->horizon < 1 || mdp->epsilon < 0.0f ||
            mdp->ng < 1 || mdp->goals == nullptr ||
            lrtdp == nullptr || lrtdp->trials < 1 || policy == nullptr) {
        fprintf(stderr, "Error[ssp_lrtdp_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = ssp_lrtdp_initialize_cpu(mdp, lrtdp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lrtdp_execute_cpu]: %s\n", "Failed to initialize the CPU variables.");
        return result;
    }

    // Iterate until you have done the desired number of trials.
    for (lrtdp->currentTrial = 0;
            lrtdp->currentTrial < lrtdp->trials && result != NOVA_CONVERGED;
            lrtdp->currentTrial++) {

        result = ssp_lrtdp_update_cpu(mdp, lrtdp);
        if (result != NOVA_SUCCESS && result != NOVA_CONVERGED) {
            fprintf(stderr, "Error[ssp_lrtdp_execute_cpu]: %s\n",
                            "Failed to perform trial of lrtdp on the CPU.");

            unsigned int resultPrime = ssp_lrtdp_uninitialize_cpu(mdp, lrtdp);
            if (resultPrime != NOVA_SUCCESS) {
                fprintf(stderr, "Error[ssp_lrtdp_execute_cpu]: %s\n",
                                "Failed to uninitialize the CPU variables.");
            }

            return result;
        }
    }

    result = ssp_lrtdp_get_policy_cpu(mdp, lrtdp, policy);
    if (result != NOVA_SUCCESS && result != NOVA_WARNING_APPROXIMATE_SOLUTION) {
        fprintf(stderr, "Error[ssp_lrtdp_execute_cpu]: %s\n", "Failed to get the policy.");

        unsigned int resultPrime = ssp_lrtdp_uninitialize_cpu(mdp, lrtdp);
        if (resultPrime != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_lrtdp_execute_cpu]: %s\n",
                            "Failed to uninitialize the CPU variables.");
        }

        return result;
    }

    bool approximateSolution = (result == NOVA_WARNING_APPROXIMATE_SOLUTION);

    result = ssp_lrtdp_uninitialize_cpu(mdp, lrtdp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lrtdp_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
    }

    // If this was an approximate solution, return this warning. Otherwise, return success.
    if (approximateSolution) {
        fprintf(stderr, "Warning[ssp_lrtdp_execute_cpu]: %s\n", "Approximate solution due to early termination and/or dead ends.");
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_SUCCESS;
}


int ssp_lrtdp_uninitialize_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp)
{
    if (mdp == nullptr || lrtdp == nullptr) {
        fprintf(stderr, "Error[ssp_lrtdp_uninitialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon and number of trials.
    lrtdp->currentHorizon = 0;
    lrtdp->currentTrial = 0;

    // Free the memory for V and pi.
    if (lrtdp->V != nullptr) {
        delete [] lrtdp->V;
    }
    lrtdp->V = nullptr;

    if (lrtdp->pi != nullptr) {
        delete [] lrtdp->pi;
    }
    lrtdp->pi = nullptr;

    return NOVA_SUCCESS;
}


int ssp_lrtdp_update_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp)
{
    // Create a visited state list (stack) variable, with just state s0.
    SSPStack visited;
    visited.maxStackSize = mdp->horizon;
    visited.stackSize = 0;
    visited.stack = nullptr;

    ssp_stack_create_cpu(visited);

    unsigned int s = mdp->s0;

    for (lrtdp->currentHorizon = 0; lrtdp->currentHorizon < mdp->horizon; lrtdp->currentHorizon++) {
        // Push this new state onto the visited stack. Break if this is absorbing.
        ssp_stack_push_cpu(visited, s);

        // Check if this is a goal or an explicit dead end.
        if (ssp_is_goal_cpu(mdp, s)) {
           lrtdp->V[s] = 0.0f;
           lrtdp->pi[s] = 0;
           break;
        } else if (ssp_is_dead_end_cpu(mdp, s)) {
            lrtdp->V[s] = NOVA_FLT_MAX;
            lrtdp->pi[s] = 0;
           break;
        }

        // Take a greedy action and update the value of this state.
        ssp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m,
                                     mdp->S, mdp->T, mdp->R,
                                     s, lrtdp->V, lrtdp->pi);

        // Randomly explore the state space using the greedy action.
        unsigned int sp = 0;
        ssp_random_successor_cpu(mdp, s, lrtdp->pi[s], sp);

        // Transition to the next state. Break if this is solved already.
        s = sp;
        if (ssp_lrtdp_is_solved_cpu(mdp, lrtdp, s)) {
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
        int result = ssp_lrtdp_check_depth_limited_solved_cpu(mdp, lrtdp, s, solved);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_lrtdp_update_cpu]: %s\n", "Failed to check solved.");
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

    if (!ssp_lrtdp_is_solved_cpu(mdp, lrtdp, mdp->s0)) {
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_CONVERGED;
}


int ssp_lrtdp_get_policy_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, MDPValueFunction *policy)
{
    if (mdp == nullptr || lrtdp == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[ssp_lrtdp_get_policy_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // First, find the best partial solution graph (nodes, which are states).
    // This is stored in the closed state list (below).

    // Create a open state list (stack).
    SSPStack open;
    open.maxStackSize = mdp->n;
    open.stackSize = 0;
    open.stack = nullptr;

    ssp_stack_create_cpu(open);
    ssp_stack_push_cpu(open, mdp->s0);

    // Create a closed state list (stack).
    SSPStack closed;
    closed.maxStackSize = mdp->n;
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
            int sp = mdp->S[s * mdp->m * mdp->ns + lrtdp->pi[s] * mdp->ns + i];
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
        fprintf(stderr, "Error[ssp_lrtdp_get_policy_cpu]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy the final (or intermediate) result, both V and pi.
    for (unsigned int i = 0; i < closed.stackSize; i++) {
        unsigned int s = closed.stack[i];
        policy->S[i] = s;
        policy->V[i] = lrtdp->V[s];
        policy->pi[i] = lrtdp->pi[s];
    }

    ssp_stack_destroy_cpu(open);
    ssp_stack_destroy_cpu(closed);

    // If the initial state is not marked as solved, then return a warning regarding the solution quality.
    if (!ssp_lrtdp_is_solved_cpu(mdp, lrtdp, mdp->s0)) {
        fprintf(stderr, "Warning[ssp_lrtdp_get_policy_cpu]: %s\n",
                "Failed to create a policy without a solved initial state.");
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

