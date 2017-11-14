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


#include <nova/mdp/algorithms/ssp_lao_star.h>

#include <nova/mdp/policies/mdp_value_function.h>
#include <nova/mdp/utilities/ssp_functions.h>
#include <nova/mdp/utilities/ssp_stack.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

#include <stdio.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <math.h>

namespace nova {

bool ssp_lao_star_is_expanded(const MDP *mdp, SSPLAOStar *lao, unsigned int s)
{
    return (lao->pi[s] < mdp->m);
}


void ssp_lao_star_reset_newly_expanded_states(const MDP *mdp, SSPLAOStar *lao)
{
    for (unsigned int s = 0; s < mdp->n; s++) {
        lao->pi[s] = mdp->m;
    }
}


bool ssp_lao_star_is_visited(const MDP *mdp, SSPLAOStar *lao, unsigned int s)
{
    return !std::signbit(lao->V[s]);
}


void ssp_lao_star_reset_visited(const MDP *mdp, SSPLAOStar *lao)
{
    for (unsigned int s = 0; s < mdp->n; s++) {
        lao->V[s] = -std::fabs(lao->V[s]);
    }
}


void ssp_lao_star_mark_visited(const MDP *mdp, SSPLAOStar *lao, unsigned int s)
{
    lao->V[s] = std::fabs(lao->V[s]);
}


void ssp_lao_star_push_successors(const MDP *mdp, SSPLAOStar *lao,
    SSPStack &stack, unsigned int s)
{
    for (unsigned int i = 0; i < mdp->ns; i++) {
        int sp = mdp->S[s * mdp->m * mdp->ns + lao->pi[s] * mdp->ns + i];
        if (sp < 0) {
            break;
        }

        // Only add it to the stack if we have not visited it yet. Putting this here
        // will ensure our stack will never overflow.
        if (!ssp_lao_star_is_visited(mdp, lao, sp)) {
            ssp_stack_push(stack, sp);
        }
    }
}


int ssp_lao_star_expand(const MDP *mdp, SSPLAOStar *lao, unsigned int &numNewlyExpandedStates)
{
    numNewlyExpandedStates = 0;

    // First, reset the visited states.
    ssp_lao_star_reset_visited(mdp, lao);

    // Create a fringe state list (stack) variable, with just state s0.
    SSPStack fringe;
    fringe.maxStackSize = lao->maxStackSize; //mdp->n;
    fringe.stackSize = 0;
    fringe.stack = nullptr;

    ssp_stack_create(fringe);
    ssp_stack_push(fringe, mdp->s0);

    // Create a traversal stack used for post order traversal Bellman updates.
    SSPStack traversal;
    traversal.maxStackSize = lao->maxStackSize; //mdp->n;
    traversal.stackSize = 0;
    traversal.stack = nullptr;

    ssp_stack_create(traversal);

    // Iterate until there are no more elements on the fringe.
    while (fringe.stackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_stack_pop(fringe, s);

        // Check if this state has been visited. If so, continue.
        if (ssp_lao_star_is_visited(mdp, lao, s)) {
            continue;
        }

        // Mark it as visited and push the state on the traversal stack for later updates.
        ssp_lao_star_mark_visited(mdp, lao, s);
        ssp_stack_push(traversal, s);

        // Check if this state is a goal. If so, continue.
        if (ssp_is_goal(mdp, s)) {
            continue;
        }

        // Only increment this if we truly have never seen this state before over any iteration of this part.
        if (!ssp_lao_star_is_expanded(mdp, lao, s)) {
            numNewlyExpandedStates++;

            // This is a newly expanded state. Perform a Bellman update.
            ssp_bellman_update(mdp->n, mdp->ns, mdp->m, mdp->S, mdp->T, mdp->R,
                                   s, lao->V, lao->pi);
        } else {
            // Otherwise, add all of its children to the fringe, as long as they are not already there, and the
            // overall set of expanded states. This follows the best action computed so far.
            ssp_lao_star_push_successors(mdp, lao, fringe, s);
        }
    }

    // At the end, in post order traversal, perform Bellman updates.
    while (traversal.stackSize != 0) {
        // Pop the node off of the traversal stack.
        unsigned int s = 0;
        ssp_stack_pop(traversal, s);

        // Perform a Bellman update on this state.
        ssp_bellman_update(mdp->n, mdp->ns, mdp->m, mdp->S, mdp->T, mdp->R,
                               s, lao->V, lao->pi);
    }

    lao->currentHorizon++;

    ssp_stack_destroy(fringe);
    ssp_stack_destroy(traversal);

    return NOVA_SUCCESS;
}


int ssp_lao_star_check_convergence(const MDP *mdp, SSPLAOStar *lao,
    bool &converged, bool &nonExpandedTipStateFound)
{
    converged = false;
    nonExpandedTipStateFound = false;

    // First, reset the visited states.
    ssp_lao_star_reset_visited(mdp, lao);

    // Create a fringe state list (stack) variable, with just state s0.
    SSPStack fringe;
    fringe.maxStackSize = lao->maxStackSize; //mdp->n;
    fringe.stackSize = 0;
    fringe.stack = nullptr;

    ssp_stack_create(fringe);
    ssp_stack_push(fringe, mdp->s0);

    // Create a traversal stack used for post order traversal Bellman updates.
    SSPStack traversal;
    traversal.maxStackSize = lao->maxStackSize; //mdp->n;
    traversal.stackSize = 0;
    traversal.stack = nullptr;

    ssp_stack_create(traversal);

    // Iterate until there are no more elements on the fringe.
    while (fringe.stackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_stack_pop(fringe, s);

        // Check if this state has been visited. If so, continue.
        if (ssp_lao_star_is_visited(mdp, lao, s)) {
            continue;
        }

        // Mark it as visited and push the state on the traversal stack for later updates.
        ssp_lao_star_mark_visited(mdp, lao, s);
        ssp_stack_push(traversal, s);

        // Check if this state is a goal. If so, continue.
        if (ssp_is_goal(mdp, s)) {
            continue;
        }

        // We have not converged if we truly have never seen this state before.
        if (!ssp_lao_star_is_expanded(mdp, lao, s)) {
            nonExpandedTipStateFound = true;
        } else {
            // Otherwise, add all of its children to the fringe, as long as they are not already there, and the
            // overall set of expanded states. This follows the best action computed so far.
            ssp_lao_star_push_successors(mdp, lao, fringe, s);
        }
    }

    // At the end, in post order traversal, perform Bellman updates. Record if the policy has changed.
    bool anActionHasChanged = false;
    float residual = 0.0f;

    while (traversal.stackSize != 0) {
        // Pop the node off of the traversal stack.
        unsigned int s = 0;
        ssp_stack_pop(traversal, s);

        float Vs = lao->V[s];
        unsigned int pis = lao->pi[s];

        // Perform a Bellman update on this state.
        ssp_bellman_update(mdp->n, mdp->ns, mdp->m, mdp->S, mdp->T, mdp->R,
                               s, lao->V, lao->pi);

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

    ssp_stack_destroy(fringe);
    ssp_stack_destroy(traversal);

    return NOVA_SUCCESS;
}


int ssp_lao_star_execute(const MDP *mdp, SSPLAOStar *lao, MDPValueFunction *policy)
{
    // First, ensure data is valid.
    if (mdp == nullptr || mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->horizon < 1 || mdp->epsilon < 0.0f ||
            mdp->ng < 1 || mdp->goals == nullptr ||
            lao == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_execute]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = ssp_lao_star_initialize(mdp, lao);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_star_execute]: %s\n", "Failed to initialize the  variables.");
        return result;
    }

    // Note: We mark a state as *never* expanded if its policy has an invalid action (i.e., index m).
    // The "expanded" list is wiped every time we expand and preserves "all valid states for the final policy".
    // Finally, the sign bit on V mark if it was visited during each internal iteration of the expand
    // function below. Confusing, but required for an efficient GPU version which will follow.
    ssp_lao_star_reset_newly_expanded_states(mdp, lao);

    // We continue the process of expanding and testing convergence until convergence occurs.
    result = NOVA_SUCCESS;
    while (result != NOVA_CONVERGED && result != NOVA_WARNING_APPROXIMATE_SOLUTION) {
        result = ssp_lao_star_update(mdp, lao);
        if (result != NOVA_SUCCESS && result != NOVA_CONVERGED
                && result != NOVA_WARNING_APPROXIMATE_SOLUTION) {
            fprintf(stderr, "Error[ssp_lao_star_execute]: %s\n",
                            "Failed to perform the update step of LAO*.");

            int resultPrime = ssp_lao_star_uninitialize(mdp, lao);
            if (resultPrime != NOVA_SUCCESS) {
                fprintf(stderr, "Error[ssp_lao_star_update]: %s\n",
                                "Failed to uninitialize the  variables.");
            }

            return result;
        }
    }

    bool approximateSolution = (result == NOVA_WARNING_APPROXIMATE_SOLUTION);

    result = ssp_lao_star_get_policy(mdp, lao, policy);
    if (result != NOVA_SUCCESS && result != NOVA_WARNING_APPROXIMATE_SOLUTION) {
        fprintf(stderr, "Error[ssp_lao_star_execute]: %s\n",
                        "Failed to get the policy.");
    }

    result = ssp_lao_star_uninitialize(mdp, lao);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_star_execute]: %s\n", "Failed to uninitialize the  variables.");
        return result;
    }

    // If this was an approximate solution, return this warning. Otherwise, return success.
    if (approximateSolution) {
        fprintf(stderr, "Warning[ssp_lao_star_execute]: %s\n", "Approximate solution due to early termination and/or dead ends.");
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_SUCCESS;
}


int ssp_lao_star_initialize(const MDP *mdp, SSPLAOStar *lao)
{
    if (mdp == nullptr || mdp->n == 0 || lao == nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_initialize]: %s\n", "Invalid arguments.");
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


int ssp_lao_star_update(const MDP *mdp, SSPLAOStar *lao)
{
    int result = 0;

    // Expand Step: Perform DFS (greedy actions) and construct a tree from possible stochastic transitions.
    // This continues until you have: (1) expanded all states, and (2) have reached one of the goal states.
    unsigned int numNewlyExpandedStates = 1;

    while (lao->currentHorizon < mdp->horizon && numNewlyExpandedStates != 0) {
        // Perform DFS (greedy actions), but mark states as visited along the way too, so it doesn't
        // revisit them. This performs a Bellman update in postorder traversal through the tree of
        // visited (plus one-level of newly expanded) nodes.
        result = ssp_lao_star_expand(mdp, lao, numNewlyExpandedStates);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_lao_star_update]: %s\n",
                            "Failed to perform expand step of LAO* on the .");
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
        result = ssp_lao_star_check_convergence(mdp, lao, converged, nonExpandedTipStateFound);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_lao_star_update]: %s\n",
                            "Failed to perform check convergence step of LAO* on the .");
            return result;
        }
    }

    // If we ran out of time, or converged without hitting a non-expanded state, then we are done.
    if (lao->currentHorizon >= mdp->horizon) {
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    } else if (converged && !nonExpandedTipStateFound) {
        return NOVA_CONVERGED;
    }

    return NOVA_SUCCESS;
}


int ssp_lao_star_get_policy(const MDP *mdp, SSPLAOStar *lao, MDPValueFunction *policy)
{
    if (mdp == nullptr || lao == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_get_policy]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // First, find the best partial solution graph (nodes, which are states).
    // This is stored in the closed state list (below).

    // Create a open state list (stack).
    SSPStack open;
    open.maxStackSize = lao->maxStackSize; //mdp->n;
    open.stackSize = 0;
    open.stack = nullptr;

    ssp_stack_create(open);
    ssp_stack_push(open, mdp->s0);

    // Create a closed state list (stack).
    SSPStack closed;
    closed.maxStackSize = lao->maxStackSize; //mdp->n;
    closed.stackSize = 0;
    closed.stack = nullptr;

    ssp_stack_create(closed);

    // Iterate until there are no more elements in the open list.
    while (open.stackSize != 0) {
        // Pop the last element off the stack.
        unsigned int s = 0;
        ssp_stack_pop(open, s);
        ssp_stack_push(closed, s);

        // If this is a goal we do not add successors.
        if (ssp_is_goal(mdp, s)) {
            continue;
        }

        // Expand the successors, following the greedy action. Add all successors
        // to the open list.
        for (unsigned int i = 0; i < mdp->ns; i++) {
            int sp = mdp->S[s * mdp->m * mdp->ns + lao->pi[s] * mdp->ns + i];
            if (sp < 0) {
                break;
            }

            // Only add to the stack if it is not already in the open or closed lists.
            if (!ssp_stack_in(open, sp) && !ssp_stack_in(closed, sp)) {
                ssp_stack_push(open, sp);
            }
        }
    }

    // Now we know how many states are in the policy. So, initialize the policy,
    // which allocates memory.
    int result = mdp_value_function_initialize(policy, mdp->n, mdp->m, closed.stackSize);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_lao_get_policy]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy the final (or intermediate) result, both V and pi.
    for (unsigned int i = 0; i < closed.stackSize; i++) {
        unsigned int s = closed.stack[i];
        policy->S[i] = s;
        policy->V[i] = lao->V[s];
        policy->pi[i] = lao->pi[s];
    }

    ssp_stack_destroy(open);
    ssp_stack_destroy(closed);

    // If the number of iterations (before it converged) is too great, then return a warning regarding the solution quality.
    if (lao->currentHorizon >= mdp->horizon) {
        fprintf(stderr, "Warning[ssp_lao_star_get_policy]: %s\n",
                "Failed to create a complete policy; maximum iterations exceeded without convergence.");
        return NOVA_WARNING_APPROXIMATE_SOLUTION;
    }

    return NOVA_SUCCESS;
}


int ssp_lao_star_uninitialize(const MDP *mdp, SSPLAOStar *lao)
{
    if (mdp == nullptr || lao == nullptr) {
        fprintf(stderr, "Error[ssp_lao_star_uninitialize]: %s\n", "Invalid arguments.");
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

}; // namespace nova

