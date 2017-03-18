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

#include <stdio.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <math.h>

#include <iostream>

#include <nova/mdp/policies/mdp_value_function.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

void ssp_lrtdp_bellman_update_cpu(unsigned int n, unsigned int ns, unsigned int m, 
    const int *S, const float *T, const float *R, unsigned int s,
    float *V, unsigned int *pi)
{
    float Vprime = NOVA_FLT_MAX;

    // Compute min_{a in A} Q(s, a). Recall, we are dealing with rewards R as positive costs.
    for (int a = 0; a < m; a++) {
        // Compute Q(s, a) for this action.
        float Qsa = R[s * m + a];

        for (int i = 0; i < ns; i++) {
            int sp = S[s * m * ns + a * ns + i];
            if (sp < 0) {
                break;
            }

            // Note: V is marked with a negative if it is solved.
            Qsa += T[s * m * ns + a * ns + i] * V[sp];
        }

        if (a == 0 || Qsa < Vprime) {
            Vprime = Qsa;
            pi[s] = a;
        }
    }

    V[s] = Vprime;
}


bool ssp_lrtdp_is_solved_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, unsigned int s)
{
    return std::signbit(lrtdp->V[s]);
}


void ssp_lrtdp_mark_solved_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, unsigned int s)
{
    lrtdp->V[s] = -std::fabs(lrtdp->V[s]);
}


void ssp_lrtdp_stack_create_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp,
    unsigned int maxStackSize, unsigned int &stackSize, unsigned int *&stack)
{
    stack = new unsigned int[maxStackSize];
    stackSize = 0;
}


void ssp_lrtdp_stack_pop_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp,
    unsigned int &stackSize, unsigned int *stack, unsigned int &s)
{
    s = stack[stackSize - 1];
    stackSize--;
}


void ssp_lrtdp_stack_push_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp,
    unsigned int &stackSize, unsigned int *stack, unsigned int s)
{
    stack[stackSize] = s;
    stackSize++;
}


bool ssp_lrtdp_stack_in_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp,
    unsigned int &stackSize, unsigned int *stack, unsigned int s)
{
    for (unsigned int i = 0; i < stackSize; i++) {
        if (stack[i] == s) {
            return true;
        }
    }
    return false;
}


void ssp_lrtdp_stack_destroy_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp,
    unsigned int &stackSize, unsigned int *&stack)
{
    delete [] stack;
    stack = nullptr;
    stackSize = 0;
}


void ssp_lrtdp_random_successor_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp,
    unsigned int s, unsigned int a, unsigned int &sp)
{
    sp = mdp->S[s * mdp->m * mdp->ns + a * mdp->ns + 0];

    float target = (float)rand() / (float)RAND_MAX;
    float current = 0.0f;

    for (unsigned int i = 0; i < mdp->ns; i++) {
        int spTmp = mdp->S[s * mdp->m * mdp->ns + a * mdp->ns + i];
        if (spTmp < 0) {
            break;
        }

        current += mdp->T[s * mdp->m * mdp->ns + a * mdp->ns + i];

        if (current >= target) {
            sp = spTmp;
            break;
        }
    }
}


bool ssp_lrtdp_is_goal_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, unsigned int s)
{
    for (unsigned int i = 0; i < mdp->ng; i++) {
        if (s == mdp->goals[i]) {
            return true;
        }
    }

    return false;
}


bool ssp_lrtdp_is_dead_end_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, unsigned int s)
{
    for (unsigned int a = 0; a < mdp->m; a++) {
        unsigned int sp = mdp->S[s * mdp->m * mdp->ns + a * mdp->ns + 0];
        float transitionProbability = mdp->T[s * mdp->m * mdp->ns + a * mdp->ns + 0];
        float cost = mdp->R[s * mdp->m + a];

        if (!(s == sp && transitionProbability == 1.0f && cost > 0.0f)) {
            return false;
        }
    }

    return true;
}


int ssp_lrtdp_check_solved_cpu(const MDP *mdp, SSPLRTDPCPU *lrtdp, bool &solved)
{
    solved = true;

    // Create a open state list (stack).
    unsigned int openStackSize = 0;
    unsigned int *openStack = nullptr;
    ssp_lrtdp_stack_create_cpu(mdp, lrtdp, mdp->n, openStackSize, openStack);

    // Create a closed state list (stack).
    unsigned int closedStackSize = 0;
    unsigned int *closedStack = nullptr;
    ssp_lrtdp_stack_create_cpu(mdp, lrtdp, mdp->n, closedStackSize, closedStack);

    // If the initial state is not solved, then push it.
    if (!ssp_lrtdp_is_solved_cpu(mdp, lrtdp, mdp->s0)) {
        ssp_lrtdp_stack_push_cpu(mdp, lrtdp, openStackSize, openStack, mdp->s0);
    }

    // Iterate until there are no more elements in the open list.
    while (openStackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_lrtdp_stack_pop_cpu(mdp, lrtdp, openStackSize, openStack, s);
        ssp_lrtdp_stack_push_cpu(mdp, lrtdp, closedStackSize, closedStack, s);

        // Perform a Bellman update on this state if it is not already solved.
        float Vs = lrtdp->V[s];
        if (!ssp_lrtdp_is_solved_cpu(mdp, lrtdp, s)) {
            ssp_lrtdp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m,
                                         mdp->S, mdp->T, mdp->R, s,
                                         lrtdp->V, lrtdp->pi);

            // This lets us compute the residual. We are not solved if it is too large.
            float residual = std::fabs(std::fabs(lrtdp->V[s]) - std::fabs(Vs));
            if (residual >= mdp->epsilon) {
                solved = false;
                continue;
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
                    && !ssp_lrtdp_stack_in_cpu(mdp, lrtdp, openStackSize, openStack, sp)
                    && !ssp_lrtdp_stack_in_cpu(mdp, lrtdp, closedStackSize, closedStack, sp)) {
                ssp_lrtdp_stack_push_cpu(mdp, lrtdp, openStackSize, openStack, sp);
            }
        }
    }

    // If this is actually solved, mark all states in the closed set as solved, too.
    // Otherwise, update all of these states.
    while (closedStackSize != 0) {
        unsigned int s = 0;
        ssp_lrtdp_stack_pop_cpu(mdp, lrtdp, closedStackSize, closedStack, s);
        if (solved) {
            ssp_lrtdp_mark_solved_cpu(mdp, lrtdp, s);
        } else {
            ssp_lrtdp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m,
                                         mdp->S, mdp->T, mdp->R, s,
                                         lrtdp->V, lrtdp->pi);
        }
    }

    ssp_lrtdp_stack_destroy_cpu(mdp, lrtdp, openStackSize, openStack);
    ssp_lrtdp_stack_destroy_cpu(mdp, lrtdp, closedStackSize, closedStack);

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
        return result;
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
    unsigned int visitedStackSize = 0;
    unsigned int *visitedStack = nullptr;
    ssp_lrtdp_stack_create_cpu(mdp, lrtdp, mdp->horizon, visitedStackSize, visitedStack);

    unsigned int s = mdp->s0;

    for (lrtdp->currentHorizon = 0; lrtdp->currentHorizon < mdp->horizon; lrtdp->currentHorizon++) {
        // Push this new state onto the visited stack. Break if this is absorbing.
        ssp_lrtdp_stack_push_cpu(mdp, lrtdp, visitedStackSize, visitedStack, s);

        // Check if this is a goal or an explicit dead end.
        if (ssp_lrtdp_is_goal_cpu(mdp, lrtdp, s)) {
           lrtdp->V[s] = 0.0f;
           lrtdp->pi[s] = 0;
           break;
        } else if (ssp_lrtdp_is_dead_end_cpu(mdp, lrtdp, s)) {
            lrtdp->V[s] = NOVA_FLT_MAX;
            lrtdp->pi[s] = 0;
           break;
        }

        // Take a greedy action and update the value of this state. Only perform
        // the Bellman update if this is not already solved.
        if (!ssp_lrtdp_is_solved_cpu(mdp, lrtdp, s)) {
            ssp_lrtdp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m,
                                         mdp->S, mdp->T, mdp->R,
                                         s, lrtdp->V, lrtdp->pi);
        }

        // Randomly explore the state space using the greedy action.
        unsigned int sp = 0;
        ssp_lrtdp_random_successor_cpu(mdp, lrtdp, s, lrtdp->pi[s], sp);

        // Transition to the next state.
        s = sp;
    }

    // At the end, in post order visited, and check for convergence ("solved").
    while (visitedStackSize != 0) {
        // Pop the node off of the traversal stack.
        unsigned int s = 0;
        ssp_lrtdp_stack_pop_cpu(mdp, lrtdp, visitedStackSize, visitedStack, s);

        // Keep checking if we "solved" these states; just return if we find one that is not.
        bool solved = false;
        int result = ssp_lrtdp_check_solved_cpu(mdp, lrtdp, solved);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_lrtdp_update_cpu]: %s\n", "Failed to check solved.");
            ssp_lrtdp_stack_destroy_cpu(mdp, lrtdp, visitedStackSize, visitedStack);
            return result;
        }

        if (!solved) {
            ssp_lrtdp_stack_destroy_cpu(mdp, lrtdp, visitedStackSize, visitedStack);
            return NOVA_SUCCESS;
        }
    }

    // If we got here, then we have solved the initial state or run out of time and are done.
    ssp_lrtdp_stack_destroy_cpu(mdp, lrtdp, visitedStackSize, visitedStack);

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
    unsigned int openStackSize = 0;
    unsigned int *openStack = nullptr;
    ssp_lrtdp_stack_create_cpu(mdp, lrtdp, mdp->n, openStackSize, openStack);
    ssp_lrtdp_stack_push_cpu(mdp, lrtdp, openStackSize, openStack, mdp->s0);

    // Create a closed state list (stack).
    unsigned int closedStackSize = 0;
    unsigned int *closedStack = nullptr;
    ssp_lrtdp_stack_create_cpu(mdp, lrtdp, mdp->n, closedStackSize, closedStack);

    // Iterate until there are no more elements in the open list.
    while (openStackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_lrtdp_stack_pop_cpu(mdp, lrtdp, openStackSize, openStack, s);
        ssp_lrtdp_stack_push_cpu(mdp, lrtdp, closedStackSize, closedStack, s);

        // If this is a goal or dead end we do not add successors.
        if (ssp_lrtdp_is_goal_cpu(mdp, lrtdp, s) || ssp_lrtdp_is_dead_end_cpu(mdp, lrtdp, s)) {
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
            if (!ssp_lrtdp_stack_in_cpu(mdp, lrtdp, openStackSize, openStack, sp)
                    && !ssp_lrtdp_stack_in_cpu(mdp, lrtdp, closedStackSize, closedStack, sp)) {
                ssp_lrtdp_stack_push_cpu(mdp, lrtdp, openStackSize, openStack, sp);
            }
        }
    }

    // Now we know how many states are in the policy. So, initialize the policy,
    // which allocates memory.
    int result = mdp_value_function_initialize(policy, mdp->n, mdp->m, closedStackSize);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lrtdp_get_policy_cpu]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy the final (or intermediate) result, both V and pi.
    for (unsigned int i = 0; i < closedStackSize; i++) {
        unsigned int s = closedStack[i];
        policy->S[i] = s;
        policy->V[i] = lrtdp->V[s];
        policy->pi[i] = lrtdp->pi[s];
    }

    ssp_lrtdp_stack_destroy_cpu(mdp, lrtdp, openStackSize, openStack);
    ssp_lrtdp_stack_destroy_cpu(mdp, lrtdp, closedStackSize, closedStack);

    // If the initial state is not marked as solved, then return a warning regarding the solution quality.
    if (!ssp_lrtdp_is_solved_cpu(mdp, lrtdp, mdp->s0)) {
        fprintf(stderr, "Warning[ssp_lrtdp_get_policy_cpu]: %s\n",
                "Failed to create a policy without a solved initial state.");
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

