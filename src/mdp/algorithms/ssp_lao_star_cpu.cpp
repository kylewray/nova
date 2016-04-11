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


#include "algorithms/ssp_lao_star_cpu.h"
#include "error_codes.h"
#include "constants.h"

#include <stdio.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <math.h>

namespace nova {

void ssp_lao_star_bellman_update_state_cpu(unsigned int n, unsigned int ns, unsigned int m, 
    const int *S, const float *T, const float *R, unsigned int s,
    float *V, unsigned int *pi)
{
    float VsPrime = FLT_MAX;

    // Compute min_{a in A} Q(s, a). Recall, we are dealing with rewards R as positive costs.
    for (int a = 0; a < m; a++) {
        // Compute Q(s, a) for this action.
        float Qsa = R[s * m + a];

        for (int i = 0; i < ns; i++) {
            int sp = S[s * m * ns + a * ns + i];
            if (sp < 0) {
                break;
            }

            // Note: V is marked with a negative based on visitation. If it had not been
            // visited, then it means it is using the heuristic value.
            Qsa += T[s * m * ns + a * ns + i] * std::fabs(V[sp]);
        }

        if (a == 0 || Qsa < VsPrime) {
            VsPrime = Qsa;
            pi[s] = a;
        }
    }

    V[s] = VsPrime;
}


bool ssp_lao_star_is_expanded(const MDP *mdp, unsigned int s)
{
    return (mdp->pi[s] < mdp->m);
}


void ssp_lao_star_reset_newly_expanded_states(MDP *mdp)
{
    for (unsigned int s = 0; s < mdp->n; s++) {
        mdp->pi[s] = mdp->m;
    }
}


bool ssp_lao_star_is_visited(const MDP *mdp, unsigned int s)
{
    return !std::signbit(mdp->V[s]);
}


void ssp_lao_star_reset_visited(MDP *mdp)
{
    for (unsigned int s = 0; s < mdp->n; s++) {
//        if (ssp_lao_star_is_visited(mdp, s)) {
            mdp->V[s] = -std::fabs(mdp->V[s]);
//        }
    }
}


void ssp_lao_star_mark_visited(const MDP *mdp, unsigned int s)
{
    mdp->V[s] = std::fabs(mdp->V[s]);
}


bool ssp_lao_star_is_goal(const MDP *mdp, unsigned int s)
{
    for (unsigned int i = 0; i < mdp->ng; i++) {
        if (s == mdp->goals[i]) {
            return true;
        }
    }
    return false;
}


void ssp_lao_star_stack_create(const MDP *mdp, unsigned int &stackSize, unsigned int *&stack)
{
    stackSize = 0;
    stack = new unsigned int[mdp->n];
}


void ssp_lao_star_stack_pop(const MDP *mdp, unsigned int &stackSize, unsigned int *stack, unsigned int &s)
{
    //s = stack[--stackSize];
    s = stack[stackSize - 1];
    stackSize--;
}


void ssp_lao_star_stack_push(const MDP *mdp, unsigned int &stackSize,
    unsigned int *stack, unsigned int s)
{
    //stack[stackSize++] = s;
    stack[stackSize] = s;
    stackSize++;
}


void ssp_lao_star_stack_push_successors(const MDP *mdp, unsigned int &stackSize,
    unsigned int *stack, unsigned int s)
{
    for (unsigned int i = 0; i < mdp->ns; i++) {
        int sp = mdp->S[s * mdp->m * mdp->ns + mdp->pi[s] * mdp->ns + i];
        if (sp < 0) {
            break;
        }

        // Only add it to the stack if we have not visited it yet. Putting this here
        // will ensure our stack will never overflow.
        if (!ssp_lao_star_is_visited(mdp, sp)) {
            ssp_lao_star_stack_push(mdp, stackSize, stack, sp);
        }
    }
}


void ssp_lao_star_stack_destroy(const MDP *mdp, unsigned int &stackSize, unsigned int *&stack)
{
    stackSize = 0;
    delete [] stack;
    stack = nullptr;
}


int ssp_lao_star_expand_cpu(MDP *mdp, unsigned int &numNewlyExpandedStates)
{
    numNewlyExpandedStates = 0;

    // First, reset the visited states.
    ssp_lao_star_reset_visited(mdp);

    // Create a fringe state list (stack) variable, with just state s0.
    unsigned int fringeStackSize = 0;
    unsigned int *fringeStack = nullptr;
    ssp_lao_star_stack_create(mdp, fringeStackSize, fringeStack);
    ssp_lao_star_stack_push(mdp, fringeStackSize, fringeStack, mdp->s0);

    // Create a traversal stack used for post order traversal Bellman updates.
    unsigned int traversalStackSize = 0;
    unsigned int *traversalStack = nullptr;
    ssp_lao_star_stack_create(mdp, traversalStackSize, traversalStack);

    // Iterate until there are no more elements on the fringe.
    while (fringeStackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_lao_star_stack_pop(mdp, fringeStackSize, fringeStack, s);

        // Check if this state has been visited. If so, continue.
        if (ssp_lao_star_is_visited(mdp, s)) {
            continue;
        }

        // Mark it as visited and push the state on the traversal stack for later updates.
        ssp_lao_star_mark_visited(mdp, s);
        ssp_lao_star_stack_push(mdp, traversalStackSize, traversalStack, s);

        // Check if this state is a goal. If so, continue.
        if (ssp_lao_star_is_goal(mdp, s)) {
            continue;
        }

        // Only increment this if we truly have never seen this state before over any iteration of this part.
        if (!ssp_lao_star_is_expanded(mdp, s)) {
            numNewlyExpandedStates++;

            // This is a newly expanded state. Perform a Bellman update.
            ssp_lao_star_bellman_update_state_cpu(mdp->n, mdp->ns, mdp->m,
                                                mdp->S, mdp->T, mdp->R, s,
                                                mdp->V, mdp->pi);
        } else {
            // Otherwise, add all of its children to the fringe, as long as they are not already there, and the
            // overall set of expanded states. This follows the best action computed so far.
            ssp_lao_star_stack_push_successors(mdp, fringeStackSize, fringeStack, s);
        }
    }

    // At the end, in post order traversal, perform Bellman updates.
    while (traversalStackSize != 0) {
        // Pop the node off of the traversal stack.
        unsigned int s = 0;
        ssp_lao_star_stack_pop(mdp, traversalStackSize, traversalStack, s);

        // Perform a Bellman update on this state.
        ssp_lao_star_bellman_update_state_cpu(mdp->n, mdp->ns, mdp->m,
                                            mdp->S, mdp->T, mdp->R, s,
                                            mdp->V, mdp->pi);
    }

    mdp->currentHorizon++;

    ssp_lao_star_stack_destroy(mdp, fringeStackSize, fringeStack);
    ssp_lao_star_stack_destroy(mdp, traversalStackSize, traversalStack);
}


int ssp_lao_star_check_convergence_cpu(MDP *mdp, bool &converged, bool &nonExpandedTipStateFound)
{
    converged = false;
    nonExpandedTipStateFound = false;

    // First, reset the visited states.
    ssp_lao_star_reset_visited(mdp);

    // Create a fringe state list (stack) variable, with just state s0.
    unsigned int fringeStackSize = 0;
    unsigned int *fringeStack = nullptr;
    ssp_lao_star_stack_create(mdp, fringeStackSize, fringeStack);
    ssp_lao_star_stack_push(mdp, fringeStackSize, fringeStack, mdp->s0);

    // Create a traversal stack used for post order traversal Bellman updates.
    unsigned int traversalStackSize = 0;
    unsigned int *traversalStack = nullptr;
    ssp_lao_star_stack_create(mdp, traversalStackSize, traversalStack);

    // Iterate until there are no more elements on the fringe.
    while (fringeStackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_lao_star_stack_pop(mdp, fringeStackSize, fringeStack, s);

        // Check if this state has been visited. If so, continue.
        if (ssp_lao_star_is_visited(mdp, s)) {
            continue;
        }

        // Mark it as visited and push the state on the traversal stack for later updates.
        ssp_lao_star_mark_visited(mdp, s);
        ssp_lao_star_stack_push(mdp, traversalStackSize, traversalStack, s);

        // Check if this state is a goal. If so, continue.
        if (ssp_lao_star_is_goal(mdp, s)) {
            continue;
        }

        // We have not converged if we truly have never seen this state before.
        if (!ssp_lao_star_is_expanded(mdp, s)) {
            nonExpandedTipStateFound = true;
        } else {
            // Otherwise, add all of its children to the fringe, as long as they are not already there, and the
            // overall set of expanded states. This follows the best action computed so far.
            ssp_lao_star_stack_push_successors(mdp, fringeStackSize, fringeStack, s);
        }
    }

    // At the end, in post order traversal, perform Bellman updates. Record if the policy has changed.
    bool anActionHasChanged = false;
    float residual = 0.0f;

    while (traversalStackSize != 0) {
        // Pop the node off of the traversal stack.
        unsigned int s = 0;
        ssp_lao_star_stack_pop(mdp, traversalStackSize, traversalStack, s);

        float Vs = mdp->V[s];
        unsigned int pis = mdp->pi[s];

        // Perform a Bellman update on this state.
        ssp_lao_star_bellman_update_state_cpu(mdp->n, mdp->ns, mdp->m,
                                            mdp->S, mdp->T, mdp->R, s,
                                            mdp->V, mdp->pi);

        // Compute the max residual.
        residual = std::max(residual, std::fabs(std::fabs(mdp->V[s]) - std::fabs(Vs)));

        // If the action changed after an update, then we have not converged.
        if (pis != mdp->pi[s]) {
            anActionHasChanged = true;
        }
    }

    if (!anActionHasChanged && residual < mdp->epsilon) {
        converged = true;
    }

    mdp->currentHorizon++;

    ssp_lao_star_stack_destroy(mdp, fringeStackSize, fringeStack);
    ssp_lao_star_stack_destroy(mdp, traversalStackSize, traversalStack);

    return NOVA_SUCCESS;
}


int ssp_lao_star_complete_cpu(MDP *mdp, const float *Vinitial, MDPValueFunction *&policy)
{
    // Note: This 'wrapper' function is provided in order to maintain 
    // the same structure as the GPU version. In the GPU version,
    // 'complete' performs the initilization and uninitialization of
    // the MDP object on the device as well. Here, we do not need that.
    return ssp_lao_star_execute_cpu(mdp, Vinitial, policy);
}


int ssp_lao_star_initialize_cpu(MDP *mdp, const float *Vinitial)
{
    // Reset the current horizon.
    mdp->currentHorizon = 0;

    // Create the variables.
    mdp->V = new float[mdp->n];
    mdp->pi = new unsigned int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    // Note that these values of V are the heuristics for each state.
    memcpy(mdp->V, Vinitial, mdp->n * sizeof(float));
    for (unsigned int i = 0; i < mdp->n; i++) {
        mdp->pi[i] = 0;
    }

    return NOVA_SUCCESS;
}


int ssp_lao_star_execute_cpu(MDP *mdp, const float *Vinitial, MDPValueFunction *&policy)
{
    int result;

    // First, ensure data is valid.
    if (mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->horizon < 1 || mdp->epsilon < 0.0f ||
            Vinitial == nullptr || policy != nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    result = ssp_lao_star_initialize_cpu(mdp, Vinitial);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n", "Failed to initialize the CPU variables.");
        return result;
    }

    // Note: We mark a state as *never* expanded if its policy has an invalid action (i.e., index m).
    // The "expanded" list is wiped every time we expand and preserves "all valid states for the final policy".
    // Finally, the sign bit on V mark if it was visited during each internal iteration of the expand
    // function below. Confusing, but required for an efficient GPU version which will follow.
    ssp_lao_star_reset_newly_expanded_states(mdp);

    // We continue the process of expanding and testing convergence until convergence occurs.
    bool running = true;
    while (running) {
        // Expand Step: Perform DFS (greedy actions) and construct a tree from possible stochastic transitions.
        // This continues until you have: (1) expanded all states, and (2) have reached one of the goal states.
        unsigned int numNewlyExpandedStates = 1;

        while (mdp->currentHorizon < mdp->horizon && numNewlyExpandedStates != 0) {
            // Perform DFS (greedy actions), but mark states as visited along the way too, so it doesn't
            // revisit them. This performs a Bellman update in postorder traversal through the tree of
            // visited (plus one-level of newly expanded) nodes.
            result = ssp_lao_star_expand_cpu(mdp, numNewlyExpandedStates);
            if (result != NOVA_SUCCESS) {
                fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n",
                                "Failed to perform expand step of LAO* on the CPU.");

                int resultPrime = ssp_lao_star_uninitialize_cpu(mdp);
                if (resultPrime != NOVA_SUCCESS) {
                    fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n",
                                    "Failed to uninitialize the CPU variables.");
                }

                return result;
            }
        }

        // Check Convergence Step: Run value iteration on expanded states until: (1) it converges (done),
        // or (2) it has an optimal action has a possible successor that was not yet expanded (continue).
        bool converged = false;
        bool nonExpandedTipStateFound = false;

        while (mdp->currentHorizon < mdp->horizon && !converged && !nonExpandedTipStateFound) {
            // Check convergence by fixing the expanded nodes and running post-order traversal of Bellman
            // update. If the policy changes, or we expand a new node, then we are not done. Otherwise,
            // check if we are within residual. If so, we have converged!
            result = ssp_lao_star_check_convergence_cpu(mdp, converged, nonExpandedTipStateFound);
            if (result != NOVA_SUCCESS) {
                fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n",
                                "Failed to perform check convergence step of LAO* on the CPU.");

                int resultPrime = ssp_lao_star_uninitialize_cpu(mdp);
                if (resultPrime != NOVA_SUCCESS) {
                    fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n",
                                    "Failed to uninitialize the CPU variables.");
                }

                return result;
            }
        }

        // If we ran out of time, or converged without hitting a non-expanded state, then we are done.
        if (mdp->currentHorizon >= mdp->horizon || (converged && !nonExpandedTipStateFound)) {
            running = false;
        }
    }

    result = ssp_lao_star_get_policy_cpu(mdp, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n", "Failed to get the policy.");

        int resultPrime = ssp_lao_star_uninitialize_cpu(mdp);
        if (resultPrime != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n",
                            "Failed to uninitialize the CPU variables.");
        }

        return result;
    }

    result = ssp_lao_star_uninitialize_cpu(mdp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int ssp_lao_star_uninitialize_cpu(MDP *mdp)
{
    // Reset the current horizon and number of expanded states.
    mdp->currentHorizon = 0;

    // Free the memory for V and pi.
    if (mdp->V != nullptr) {
        delete [] mdp->V;
    }
    mdp->V = nullptr;

    if (mdp->pi != nullptr) {
        delete [] mdp->pi;
    }
    mdp->pi = nullptr;

    return NOVA_SUCCESS;
}


int ssp_lao_star_get_policy_cpu(const MDP *mdp, MDPValueFunction *&policy)
{
    if (policy != nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_get_policy_cpu]: %s\n",
                        "Invalid arguments. The policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    policy = new MDPValueFunction();

    policy->n = mdp->n;
    policy->m = mdp->m;

    // First, count the number of states that are valid following the policy.
    policy->r = 0;

    for (unsigned int s = 0; s < mdp->n; s++) {
        // Only include the visited states as part of the final policy. These states are all reachable states
        // following the optimal policy. Recall that some states might be expanded early on in the process,
        // but are quickly abandoned.
        if (ssp_lao_star_is_visited(mdp, s)) {
            policy->r++;
        }
    }

    policy->S = new unsigned int[policy->r];
    policy->V = new float[policy->r];
    policy->pi = new unsigned int[policy->r];

    // Copy the final (or intermediate) result, both V and pi. This assumes memory has been allocated
    // for the variables provided. Importantly, only the values of the visited states are copied.
    // The non-visited states are left alone.
    unsigned int r = 0;

    for (unsigned int s = 0; s < mdp->n; s++) {
        // Only include the visited states as part of the final policy. These states are all reachable states
        // following the optimal policy. Recall that some states might be expanded early on in the process,
        // but are quickly abandoned.
        if (ssp_lao_star_is_visited(mdp, s)) {
            policy->S[r] = s;
            policy->V[r] = mdp->V[s];
            policy->pi[r] = mdp->pi[s];
            r++;
        }
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

