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


#include <nova/mdp/algorithms/ssp_lao_star_cpu.h>

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

void ssp_lao_star_bellman_update_cpu(unsigned int n, unsigned int ns, unsigned int m, 
    const int *S, const float *T, const float *R, unsigned int s,
    float *V, unsigned int *pi)
{
    float VsPrime = NOVA_FLT_MAX;

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


bool ssp_lao_star_is_expanded_cpu(const MDP *mdp, SSPLAOStarCPU *lao, unsigned int s)
{
    return (lao->pi[s] < mdp->m);
}


void ssp_lao_star_reset_newly_expanded_states_cpu(const MDP *mdp, SSPLAOStarCPU *lao)
{
    for (unsigned int s = 0; s < mdp->n; s++) {
        lao->pi[s] = mdp->m;
    }
}


bool ssp_lao_star_is_visited_cpu(const MDP *mdp, SSPLAOStarCPU *lao, unsigned int s)
{
    return !std::signbit(lao->V[s]);
}


void ssp_lao_star_reset_visited_cpu(const MDP *mdp, SSPLAOStarCPU *lao)
{
    for (unsigned int s = 0; s < mdp->n; s++) {
        lao->V[s] = -std::fabs(lao->V[s]);
    }
}


void ssp_lao_star_mark_visited_cpu(const MDP *mdp, SSPLAOStarCPU *lao, unsigned int s)
{
    lao->V[s] = std::fabs(lao->V[s]);
}


bool ssp_lao_star_is_goal_cpu(const MDP *mdp, SSPLAOStarCPU *lao, unsigned int s)
{
    for (unsigned int i = 0; i < mdp->ng; i++) {
        if (s == mdp->goals[i]) {
            return true;
        }
    }
    return false;
}


void ssp_lao_star_stack_create_cpu(const MDP *mdp, SSPLAOStarCPU *lao,
    unsigned int &stackSize, unsigned int *&stack)
{
    stack = new unsigned int[mdp->n];
    stackSize = 0;
}


void ssp_lao_star_stack_pop_cpu(const MDP *mdp, SSPLAOStarCPU *lao,
    unsigned int &stackSize, unsigned int *stack, unsigned int &s)
{
    s = stack[stackSize - 1];
    stackSize--;
}


void ssp_lao_star_stack_push_cpu(const MDP *mdp, SSPLAOStarCPU *lao,
    unsigned int &stackSize, unsigned int *stack, unsigned int s)
{
    stack[stackSize] = s;
    stackSize++;
}


void ssp_lao_star_stack_push_successors_cpu(const MDP *mdp, SSPLAOStarCPU *lao,
    unsigned int &stackSize, unsigned int *stack, unsigned int s)
{
    for (unsigned int i = 0; i < mdp->ns; i++) {
        int sp = mdp->S[s * mdp->m * mdp->ns + lao->pi[s] * mdp->ns + i];
        if (sp < 0) {
            break;
        }

        // Only add it to the stack if we have not visited it yet. Putting this here
        // will ensure our stack will never overflow.
        if (!ssp_lao_star_is_visited_cpu(mdp, lao, sp)) {
            ssp_lao_star_stack_push_cpu(mdp, lao, stackSize, stack, sp);
        }
    }
}


void ssp_lao_star_stack_destroy_cpu(const MDP *mdp, SSPLAOStarCPU *lao,
    unsigned int &stackSize, unsigned int *&stack)
{
    delete [] stack;
    stack = nullptr;
    stackSize = 0;
}


int ssp_lao_star_expand_cpu(const MDP *mdp, SSPLAOStarCPU *lao,
    unsigned int &numNewlyExpandedStates)
{
    numNewlyExpandedStates = 0;

    // First, reset the visited states.
    ssp_lao_star_reset_visited_cpu(mdp, lao);

    // Create a fringe state list (stack) variable, with just state s0.
    unsigned int fringeStackSize = 0;
    unsigned int *fringeStack = nullptr;
    ssp_lao_star_stack_create_cpu(mdp, lao, fringeStackSize, fringeStack);
    ssp_lao_star_stack_push_cpu(mdp, lao, fringeStackSize, fringeStack, mdp->s0);

    // Create a traversal stack used for post order traversal Bellman updates.
    unsigned int traversalStackSize = 0;
    unsigned int *traversalStack = nullptr;
    ssp_lao_star_stack_create_cpu(mdp, lao, traversalStackSize, traversalStack);

    // Iterate until there are no more elements on the fringe.
    while (fringeStackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_lao_star_stack_pop_cpu(mdp, lao, fringeStackSize, fringeStack, s);

        // Check if this state has been visited. If so, continue.
        if (ssp_lao_star_is_visited_cpu(mdp, lao, s)) {
            continue;
        }

        // Mark it as visited and push the state on the traversal stack for later updates.
        ssp_lao_star_mark_visited_cpu(mdp, lao, s);
        ssp_lao_star_stack_push_cpu(mdp, lao, traversalStackSize, traversalStack, s);

        // Check if this state is a goal. If so, continue.
        if (ssp_lao_star_is_goal_cpu(mdp, lao, s)) {
            continue;
        }

        // Only increment this if we truly have never seen this state before over any iteration of this part.
        if (!ssp_lao_star_is_expanded_cpu(mdp, lao, s)) {
            numNewlyExpandedStates++;

            // This is a newly expanded state. Perform a Bellman update.
            ssp_lao_star_bellman_update_cpu(mdp->n, mdp->ns, mdp->m,
                                                mdp->S, mdp->T, mdp->R, s,
                                                lao->V, lao->pi);
        } else {
            // Otherwise, add all of its children to the fringe, as long as they are not already there, and the
            // overall set of expanded states. This follows the best action computed so far.
            ssp_lao_star_stack_push_successors_cpu(mdp, lao, fringeStackSize, fringeStack, s);
        }
    }

    // At the end, in post order traversal, perform Bellman updates.
    while (traversalStackSize != 0) {
        // Pop the node off of the traversal stack.
        unsigned int s = 0;
        ssp_lao_star_stack_pop_cpu(mdp, lao, traversalStackSize, traversalStack, s);

        // Perform a Bellman update on this state.
        ssp_lao_star_bellman_update_cpu(mdp->n, mdp->ns, mdp->m,
                                        mdp->S, mdp->T, mdp->R, s,
                                        lao->V, lao->pi);
    }

    lao->currentHorizon++;

    ssp_lao_star_stack_destroy_cpu(mdp, lao, fringeStackSize, fringeStack);
    ssp_lao_star_stack_destroy_cpu(mdp, lao, traversalStackSize, traversalStack);

    return NOVA_SUCCESS;
}


int ssp_lao_star_check_convergence_cpu(const MDP *mdp, SSPLAOStarCPU *lao,
    bool &converged, bool &nonExpandedTipStateFound)
{
    converged = false;
    nonExpandedTipStateFound = false;

    // First, reset the visited states.
    ssp_lao_star_reset_visited_cpu(mdp, lao);

    // Create a fringe state list (stack) variable, with just state s0.
    unsigned int fringeStackSize = 0;
    unsigned int *fringeStack = nullptr;
    ssp_lao_star_stack_create_cpu(mdp, lao, fringeStackSize, fringeStack);
    ssp_lao_star_stack_push_cpu(mdp, lao, fringeStackSize, fringeStack, mdp->s0);

    // Create a traversal stack used for post order traversal Bellman updates.
    unsigned int traversalStackSize = 0;
    unsigned int *traversalStack = nullptr;
    ssp_lao_star_stack_create_cpu(mdp, lao, traversalStackSize, traversalStack);

    // Iterate until there are no more elements on the fringe.
    while (fringeStackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_lao_star_stack_pop_cpu(mdp, lao, fringeStackSize, fringeStack, s);

        // Check if this state has been visited. If so, continue.
        if (ssp_lao_star_is_visited_cpu(mdp, lao, s)) {
            continue;
        }

        // Mark it as visited and push the state on the traversal stack for later updates.
        ssp_lao_star_mark_visited_cpu(mdp, lao, s);
        ssp_lao_star_stack_push_cpu(mdp, lao, traversalStackSize, traversalStack, s);

        // Check if this state is a goal. If so, continue.
        if (ssp_lao_star_is_goal_cpu(mdp, lao, s)) {
            continue;
        }

        // We have not converged if we truly have never seen this state before.
        if (!ssp_lao_star_is_expanded_cpu(mdp, lao, s)) {
            nonExpandedTipStateFound = true;
        } else {
            // Otherwise, add all of its children to the fringe, as long as they are not already there, and the
            // overall set of expanded states. This follows the best action computed so far.
            ssp_lao_star_stack_push_successors_cpu(mdp, lao, fringeStackSize, fringeStack, s);
        }
    }

    // At the end, in post order traversal, perform Bellman updates. Record if the policy has changed.
    bool anActionHasChanged = false;
    float residual = 0.0f;

    while (traversalStackSize != 0) {
        // Pop the node off of the traversal stack.
        unsigned int s = 0;
        ssp_lao_star_stack_pop_cpu(mdp, lao, traversalStackSize, traversalStack, s);

        float Vs = lao->V[s];
        unsigned int pis = lao->pi[s];

        // Perform a Bellman update on this state.
        ssp_lao_star_bellman_update_cpu(mdp->n, mdp->ns, mdp->m,
                                        mdp->S, mdp->T, mdp->R, s,
                                        lao->V, lao->pi);

        // Compute the max residual.
        residual = std::max(residual, std::fabs(std::fabs(lao->V[s]) - std::fabs(Vs)));

        // If the action changed after an update, then we have not converged.
        if (pis != lao->pi[s]) {
            anActionHasChanged = true;
        }
    }

    if (!anActionHasChanged && residual < mdp->epsilon) {
        converged = true;
    }

    lao->currentHorizon++;

    ssp_lao_star_stack_destroy_cpu(mdp, lao, fringeStackSize, fringeStack);
    ssp_lao_star_stack_destroy_cpu(mdp, lao, traversalStackSize, traversalStack);

    return NOVA_SUCCESS;
}


int ssp_lao_star_initialize_cpu(const MDP *mdp, SSPLAOStarCPU *lao)
{
    if (mdp == nullptr || mdp->n == 0 || lao == nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_initialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon.
    lao->currentHorizon = 0;

    // Create the variables.
    lao->V = new float[mdp->n];
    lao->pi = new unsigned int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    // If undefined, then assign 0 for V.
    if (lao->VInitial != nullptr) {
        memcpy(lao->V, lao->VInitial, mdp->n * sizeof(float));
        for (unsigned int i = 0; i < mdp->n; i++) {
            lao->pi[i] = 0;
        }
    } else {
        for (unsigned int i = 0; i < mdp->n; i++) {
            lao->V[i] = 0.0f;
            lao->pi[i] = 0;
        }
    }

    return NOVA_SUCCESS;
}


int ssp_lao_star_execute_cpu(const MDP *mdp, SSPLAOStarCPU *lao, MDPValueFunction *policy)
{
    // First, ensure data is valid.
    if (mdp == nullptr || mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->horizon < 1 || mdp->epsilon < 0.0f ||
            mdp->ng < 1 || mdp->goals == nullptr ||
            lao == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = ssp_lao_star_initialize_cpu(mdp, lao);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n", "Failed to initialize the CPU variables.");
        return result;
    }

    // Note: We mark a state as *never* expanded if its policy has an invalid action (i.e., index m).
    // The "expanded" list is wiped every time we expand and preserves "all valid states for the final policy".
    // Finally, the sign bit on V mark if it was visited during each internal iteration of the expand
    // function below. Confusing, but required for an efficient GPU version which will follow.
    ssp_lao_star_reset_newly_expanded_states_cpu(mdp, lao);

    // We continue the process of expanding and testing convergence until convergence occurs.
    result = NOVA_SUCCESS;
    while (result != NOVA_CONVERGED) {
        result = ssp_lao_star_update_cpu(mdp, lao);
        if (result != NOVA_SUCCESS && result != NOVA_CONVERGED) {
            fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n",
                        "Failed to perform the update step of LAO*.");

            int resultPrime = ssp_lao_star_uninitialize_cpu(mdp, lao);
            if (resultPrime != NOVA_SUCCESS) {
                fprintf(stderr, "Error[ssp_lao_star_update_cpu]: %s\n",
                                "Failed to uninitialize the CPU variables.");
            }

            return result;
        }
    }

    result = ssp_lao_star_get_policy_cpu(mdp, lao, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n", "Failed to get the policy.");

        int resultPrime = ssp_lao_star_uninitialize_cpu(mdp, lao);
        if (resultPrime != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n",
                            "Failed to uninitialize the CPU variables.");
        }

        return result;
    }

    result = ssp_lao_star_uninitialize_cpu(mdp, lao);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_star_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int ssp_lao_star_uninitialize_cpu(const MDP *mdp, SSPLAOStarCPU *lao)
{
    if (mdp == nullptr || lao == nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_uninitialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon and number of expanded states.
    lao->currentHorizon = 0;

    // Free the memory for V and pi.
    if (lao->V != nullptr) {
        delete [] lao->V;
    }
    lao->V = nullptr;

    if (lao->pi != nullptr) {
        delete [] lao->pi;
    }
    lao->pi = nullptr;

    return NOVA_SUCCESS;
}


int ssp_lao_star_update_cpu(const MDP *mdp, SSPLAOStarCPU *lao)
{
    int result = 0;

    // Expand Step: Perform DFS (greedy actions) and construct a tree from possible stochastic transitions.
    // This continues until you have: (1) expanded all states, and (2) have reached one of the goal states.
    unsigned int numNewlyExpandedStates = 1;

    while (lao->currentHorizon < mdp->horizon && numNewlyExpandedStates != 0) {
        // Perform DFS (greedy actions), but mark states as visited along the way too, so it doesn't
        // revisit them. This performs a Bellman update in postorder traversal through the tree of
        // visited (plus one-level of newly expanded) nodes.
        result = ssp_lao_star_expand_cpu(mdp, lao, numNewlyExpandedStates);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_lao_star_update_cpu]: %s\n",
                            "Failed to perform expand step of LAO* on the CPU.");
            return result;
        }
    }

    // Check Convergence Step: Run value iteration on expanded states until: (1) it converges (done),
    // or (2) it has an optimal action has a possible successor that was not yet expanded (continue).
    bool converged = false;
    bool nonExpandedTipStateFound = false;

    while (lao->currentHorizon < mdp->horizon && !converged && !nonExpandedTipStateFound) {
        // Check convergence by fixing the expanded nodes and running post-order traversal of Bellman
        // update. If the policy changes, or we expand a new node, then we are not done. Otherwise,
        // check if we are within residual. If so, we have converged!
        result = ssp_lao_star_check_convergence_cpu(mdp, lao, converged, nonExpandedTipStateFound);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_lao_star_update_cpu]: %s\n",
                            "Failed to perform check convergence step of LAO* on the CPU.");
            return result;
        }
    }

    // If we ran out of time, or converged without hitting a non-expanded state, then we are done.
    if (lao->currentHorizon >= mdp->horizon || (converged && !nonExpandedTipStateFound)) {
        return NOVA_CONVERGED;
    }

    return NOVA_SUCCESS;
}


int ssp_lao_star_get_policy_cpu(const MDP *mdp, SSPLAOStarCPU *lao, MDPValueFunction *policy)
{
    if (mdp == nullptr || lao == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_get_policy_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // First, count the number of states that are valid following the policy.
    unsigned int r = 0;

    for (unsigned int s = 0; s < mdp->n; s++) {
        // Only include the visited states as part of the final policy. These states are all reachable states
        // following the optimal policy. Recall that some states might be expanded early on in the process,
        // but are quickly abandoned.
        if (ssp_lao_star_is_visited_cpu(mdp, lao, s)) {
            r++;
        }
    }

    // Initialize the policy, which allocates memory.
    int result = mdp_value_function_initialize(policy, mdp->n, mdp->m, r);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_star_get_policy_cpu]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy the final (or intermediate) result, both V and pi. This assumes memory has been allocated
    // for the variables provided. Importantly, only the values of the visited states are copied.
    // The non-visited states are left alone.
    r = 0;

    for (unsigned int s = 0; s < mdp->n; s++) {
        // Only include the visited states as part of the final policy. These states are all reachable states
        // following the optimal policy. Recall that some states might be expanded early on in the process,
        // but are quickly abandoned.
        if (ssp_lao_star_is_visited_cpu(mdp, lao, s)) {
            policy->S[r] = s;
            policy->V[r] = lao->V[s];
            policy->pi[r] = lao->pi[s];
            r++;
        }
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

