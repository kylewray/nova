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
    float Vs = FLT_MAX;

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

        if (a == 0 || Qsa < Vs) {
            Vs = Qsa;
            pi[s] = a;
        }
    }

    V[s] = Vs;
}


int ssp_rtdp_complete_cpu(MDP *mdp, const float *Vinitial, MDPValueFunction *&policy)
{
    // Note: This 'wrapper' function is provided in order to maintain 
    // the same structure as the GPU version. In the GPU version,
    // 'complete' performs the initilization and uninitialization of
    // the MDP object on the device as well. Here, we do not need that.
    return ssp_rtdp_execute_cpu(mdp, Vinitial, policy);
}


int ssp_rtdp_initialize_cpu(MDP *mdp, const float *Vinitial)
{
    // Reset the current horizon.
    mdp->currentHorizon = 0;

    // Create the variables.
    mdp->V = new float[mdp->n];
    mdp->pi = new unsigned int[mdp->n];

    mdp->ne = 0;
    mdp->expanded = new int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    // Note that these values of V are the heuristics for each state.

    // Additionally, the default values for pi are m, which is an
    // impossible action. This trick enables us to know if we have
    // ever visited a state or not, and thus know when to add a newly
    // expanded state to the set of expanded states.

    // Also, the default values for the expanded states are -1, meaning
    // no expanded state is defined for the index. Goals must have a
    // value of 0, too.
    memcpy(mdp->V, Vinitial, mdp->n * sizeof(float));

    for (unsigned int i = 0; i < mdp->ng; i++) {
        mdp->V[mdp->goals[i]] = 0.0f;
    }

    for (unsigned int i = 0; i < mdp->n; i++) {
        mdp->pi[i] = mdp->m;
        mdp->expanded[i] = -1;
    }

    return NOVA_SUCCESS;
}


int ssp_rtdp_execute_cpu(MDP *mdp, const float *Vinitial, MDPValueFunction *&policy)
{
    int result;

    // First, ensure data is valid.
    if (mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->horizon < 1 || mdp->epsilon < 0.0f ||
            mdp->ne != 0 || mdp->expanded != nullptr ||
            Vinitial == nullptr || policy != nullptr) {
        fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    result = ssp_rtdp_initialize_cpu(mdp, Vinitial);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Failed to initialize the CPU variables.");
        return result;
    }

    // Iterate until you have done 'horizon' trials of RTDP.
    while (mdp->currentHorizon < mdp->horizon) {
        result = ssp_rtdp_update_cpu(mdp);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Failed to perform trial of RTDP on the CPU.");
            return result;
        }
    }

    result = ssp_rtdp_get_policy_cpu(mdp, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Failed to get the policy.");
        return result;
    }

    result = ssp_rtdp_uninitialize_cpu(mdp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[ssp_rtdp_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int ssp_rtdp_update_cpu(MDP *mdp)
{
    unsigned int s = mdp->s0;
    bool isGoal = false;
    bool isDeadEnd = false;

    while (!isGoal && !isDeadEnd) {
        // Take a greedy action and update the value of this state. We oscillate between V depending on the step.
        ssp_rtdp_bellman_update_state_cpu(mdp->n, mdp->ns, mdp->m,
                                            mdp->S, mdp->T, mdp->R,
                                            s, mdp->V, mdp->pi);

        // This is the greedy action.
        unsigned int a = mdp->pi[s];

        // Randomly explore the state space using the action.
        float target = (float)rand() / (float)RAND_MAX;
        float current = 0.0f;
        unsigned int sp = mdp->m;

        for (unsigned int i = 0; i < mdp->ns; i++) {
            int spTmp = mdp->S[s * mdp->m * mdp->ns + a * mdp->ns + i];
            if (spTmp < 0) {
                break;
            }

            // For any strange edge cases, we ensure a valid state
            // transition can arise.
            if (sp == mdp->m) {
                sp = spTmp;
            }

            current += mdp->T[s * mdp->m * mdp->ns + a * mdp->ns + i];

            if (current >= target) {
                sp = spTmp;
                break;
            }
        }

        s = sp;

        // Add s to the set of expanded states, and set the action to
        // a valid one. This assignment ensures that goal states will
        // have a valid action, with their default values being invalid.
        if (mdp->pi[s] == mdp->m) {
            mdp->pi[s] = 0;

            mdp->expanded[mdp->ne] = s;
            mdp->ne++;
        }

        // Check if s is a goal.
        for (unsigned int i = 0; i < mdp->ng; i++) {
            if (s == mdp->goals[i]) {
                mdp->V[s] = 0.0f;
                isGoal = true;
                break;
            }
        }

        //*
        // ***Special Modification***
        // If this is an explicit dead end, meaning non-zero cost for
        // all actions and it is a self-loop, then we terminate but
        // assign the value of V to be maximal possible to within
        // machine precision.
        isDeadEnd = true;
        for (unsigned int ap = 0; ap < mdp->m; ap++) {
            if (!(mdp->T[s * mdp->m * mdp->ns + ap * mdp->ns + 0] == 1.0f && mdp->R[s * mdp->m + ap] > 0.0f)) {
                isDeadEnd = false;
                break;
            }
        }

        if (isDeadEnd) {
            mdp->V[s] = FLT_MAX;
        }
        //*/
    }

    mdp->currentHorizon++;

    return NOVA_SUCCESS;
}


int ssp_rtdp_uninitialize_cpu(MDP *mdp)
{
    // Reset the current horizon and number of expanded states.
    mdp->currentHorizon = 0;
    mdp->ne = 0;

    // Free the expanded states set.
    if (mdp->expanded != nullptr) {
        delete [] mdp->expanded;
    }
    mdp->expanded = nullptr;

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


int ssp_rtdp_get_policy_cpu(const MDP *mdp, MDPValueFunction *&policy)
{
    if (policy != nullptr) {
        fprintf(stderr, "Error[ssp_rtdp_get_policy_cpu]: %s\n", "Invalid arguments. The policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    policy = new MDPValueFunction();

    policy->n = mdp->n;
    policy->m = mdp->m;
    policy->r = mdp->ne;

    policy->S = new unsigned int[policy->r];
    policy->V = new float[policy->r];
    policy->pi = new unsigned int[policy->r];

    // Copy the final (or intermediate) result, both V and pi.
    for (unsigned int i = 0; i < mdp->ne; i++) {
        unsigned int s = mdp->expanded[i];

        policy->S[i] = s;
        policy->V[i] = mdp->V[s];
        policy->pi[i] = mdp->pi[s];
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

