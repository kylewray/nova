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


#include "algorithms/ssp_rtdp_cpu.h"
#include "error_codes.h"
#include "constants.h"

#include <stdio.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <math.h>

namespace nova {

void ssp_rtdp_bellman_update_state_cpu(unsigned int n, unsigned int ns, unsigned int m, 
                            const int *S, const float *T, const float *R,
                            unsigned int s, float *V, unsigned int *pi)
{
    float Vprime = FLT_MAX;

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
            Qsa += T[s * m * ns + a * ns + i] * V[sp];
        }

        if (a == 0 || Qsa < Vprime) {
            Vprime = Qsa;
            pi[s] = a;
        }
    }

    V[s] = Vprime;
}


bool ssp_rtdp_is_expanded(const MDP *mdp, SSPRTDPCPU *rtdp, unsigned int s)
{
    return (rtdp->pi[s] < mdp->m);
}


void ssp_rtdp_random_successor(const MDP *mdp, SSPRTDPCPU *rtdp,
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


bool ssp_rtdp_is_goal(const MDP *mdp, SSPRTDPCPU *rtdp, unsigned int s)
{
    for (unsigned int i = 0; i < mdp->ng; i++) {
        if (s == mdp->goals[i]) {
            return true;
        }
    }

    return false;
}


bool ssp_rtdp_is_dead_end(const MDP *mdp, SSPRTDPCPU *rtdp, unsigned int s)
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


int ssp_rtdp_initialize_cpu(const MDP *mdp, SSPRTDPCPU *rtdp)
{
    // Reset the current trial and horizon.
    rtdp->currentTrial = 0;
    rtdp->currentHorizon = 0;

    // Create the variables.
    rtdp->V = new float[mdp->n];
    rtdp->pi = new unsigned int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    // Note that these values of V are the heuristics for each state.
    // Also, we use pi to determine expanded states. If pi has value m,
    // then it is not expanded. Else if it is a valid action in
    // {0, ..., m-1}, then it is expanded.
    memcpy(rtdp->V, rtdp->Vinitial, mdp->n * sizeof(float));
    for (unsigned int i = 0; i < mdp->n; i++) {
        rtdp->pi[i] = mdp->m;
    }

    return NOVA_SUCCESS;
}


int ssp_rtdp_execute_cpu(const MDP *mdp, SSPRTDPCPU *rtdp, MDPValueFunction *&policy)
{
    int result;

    // First, ensure data is valid.
    if (mdp == nullptr || mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->horizon < 1 || mdp->epsilon < 0.0f ||
            rtdp == nullptr || rtdp->Vinitial == nullptr || policy != nullptr) {
        fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    result = ssp_rtdp_initialize_cpu(mdp, rtdp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Failed to initialize the CPU variables.");
        return result;
    }

    // Iterate until you have done the desired number of trials.
    for (rtdp->currentTrial = 0; rtdp->currentTrial < rtdp->trials; rtdp->currentTrial++) {
        result = ssp_rtdp_update_cpu(mdp, rtdp);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Failed to perform trial of RTDP on the CPU.");

            unsigned int resultPrime = ssp_rtdp_uninitialize_cpu(mdp, rtdp);
            if (resultPrime != NOVA_SUCCESS) {
                fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n",
                                "Failed to uninitialize the CPU variables.");
            }

            return result;
        }
    }

    result = ssp_rtdp_get_policy_cpu(mdp, rtdp, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Failed to get the policy.");

        unsigned int resultPrime = ssp_rtdp_uninitialize_cpu(mdp, rtdp);
        if (resultPrime != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n",
                            "Failed to uninitialize the CPU variables.");
        }

        return result;
    }

    result = ssp_rtdp_uninitialize_cpu(mdp, rtdp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int ssp_rtdp_update_cpu(const MDP *mdp, SSPRTDPCPU *rtdp)
{
    unsigned int s = mdp->s0;
    bool isGoal = ssp_rtdp_is_goal(mdp, rtdp, s);
    bool isDeadEnd = false; //ssp_rtdp_is_dead_end(mdp, s);

    rtdp->currentHorizon = 0;

    while (!isGoal && !isDeadEnd && rtdp->currentHorizon < mdp->horizon) {
        // Take a greedy action and update the value of this state.
        ssp_rtdp_bellman_update_state_cpu(mdp->n, mdp->ns, mdp->m,
                                            mdp->S, mdp->T, mdp->R,
                                            s, rtdp->V, rtdp->pi);

        // This is the greedy action.
        unsigned int a = rtdp->pi[s];

        // Randomly explore the state space using the action.
        unsigned int sp = 0;
        ssp_rtdp_random_successor(mdp, rtdp, s, a, sp);

        // Transition to the next state.
        s = sp;

        // Check if this is a goal or an explicit dead end. If so, then we will stop.
        isGoal = ssp_rtdp_is_goal(mdp, rtdp, s);
        isDeadEnd = ssp_rtdp_is_dead_end(mdp, rtdp, s);

        if (isGoal) {
            rtdp->V[s] = 0.0f;
        }
        if (isDeadEnd) {
            rtdp->V[s] = FLT_MAX;
        }

        rtdp->currentHorizon++;
    }

    return NOVA_SUCCESS;
}


int ssp_rtdp_uninitialize_cpu(const MDP *mdp, SSPRTDPCPU *rtdp)
{
    // Reset the current horizon and number of trials.
    rtdp->currentHorizon = 0;
    rtdp->currentTrial = 0;

    // Free the memory for V and pi.
    if (rtdp->V != nullptr) {
        delete [] rtdp->V;
    }
    rtdp->V = nullptr;

    if (rtdp->pi != nullptr) {
        delete [] rtdp->pi;
    }
    rtdp->pi = nullptr;

    return NOVA_SUCCESS;
}


int ssp_rtdp_get_policy_cpu(const MDP *mdp, SSPRTDPCPU *rtdp, MDPValueFunction *&policy)
{
    if (policy != nullptr) {
        fprintf(stderr, "Error[ssp_rtdp_get_policy_cpu]: %s\n",
                        "Invalid arguments. The policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    policy = new MDPValueFunction();

    policy->n = mdp->n;
    policy->m = mdp->m;

    // First, count the number of states that are valid following the policy.
    policy->r = 0;

    for (unsigned int s = 0; s < mdp->n; s++) {
        // Only include the expanded (visited) states as part of the final policy.
        if (ssp_rtdp_is_expanded(mdp, rtdp, s)) {
            policy->r++;
        }
    }

    policy->S = new unsigned int[policy->r];
    policy->V = new float[policy->r];
    policy->pi = new unsigned int[policy->r];

    // Copy the final (or intermediate) result, both V and pi.
    unsigned int r = 0;

    for (unsigned int s = 0; s < mdp->n; s++) {
        if (ssp_rtdp_is_expanded(mdp, rtdp, s)) {
            policy->S[r] = s;
            policy->V[r] = rtdp->V[s];
            policy->pi[r] = rtdp->pi[s];
            r++;
        }
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

