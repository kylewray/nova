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


void ssp_rtdp_bellman_update_state_cpu(unsigned int n, unsigned int ns, unsigned int m, 
                            const int *S, const float *T, const float *R, const float *V,
                            unsigned int ne, const int *expanded,
                            unsigned int s,
                            float *VPrime, unsigned int *pi)
{
    VPrime[s] = FLT_MAX;

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

        if (a == 0 || Qsa < VPrime[s]) {
            VPrime[s] = Qsa;
            pi[s] = a;
        }
    }
}


int ssp_rtdp_complete_cpu(MDP *mdp, const float *Vinitial, unsigned int &r, unsigned int *&S, float *&V, unsigned int *&pi)
{
    // Note: This 'wrapper' function is provided in order to maintain 
    // the same structure as the GPU version. In the GPU version,
    // 'complete' performs the initilization and uninitialization of
    // the MDP object on the device as well. Here, we do not need that.
    return ssp_rtdp_execute_cpu(mdp, Vinitial, r, S, V, pi);
}


int ssp_rtdp_initialize_cpu(MDP *mdp, const float *Vinitial)
{
    // Reset the current horizon.
    mdp->currentHorizon = 0;

    // Create the variables.
    mdp->V = new float[mdp->n];
    mdp->VPrime = new float[mdp->n];
    mdp->pi = new unsigned int[mdp->n];

    mdp->ne = 0;
    mdp->expanded = new int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    // Note that these values of V are the heuristics for each state.
    // Also, the default values for the expanded states are -1, meaning
    // no expanded state is defined for the index.
    memcpy(mdp->V, Vinitial, mdp->n * sizeof(float));
    memcpy(mdp->VPrime, Vinitial, mdp->n * sizeof(float));
    for (unsigned int i = 0; i < mdp->n; i++) {
        mdp->pi[i] = 0;
        mdp->expanded[i] = -1;
    }

    return NOVA_SUCCESS;
}


int ssp_rtdp_execute_cpu(MDP *mdp, const float *Vinitial, unsigned int &r, unsigned int *&S, float *&V, unsigned int *&pi)
{
    int result;

    // First, ensure data is valid.
    if (mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->horizon < 1 || mdp->epsilon < 0.0f ||
            //mdp->ne != 0 || mdp->expanded == nullptr ||
            Vinitial == nullptr || S != nullptr || V != nullptr || pi != nullptr) {
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

    result = ssp_rtdp_get_policy_cpu(mdp, r, S, V, pi);
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
    bool sIsGoal = false;

    while (!sIsGoal) {
        // Take a greedy action and update the value of this state. We oscillate between V and VPrime depending on the step.
        if (mdp->currentHorizon % 2 == 0) {
            ssp_rtdp_bellman_update_state_cpu(mdp->n, mdp->ns, mdp->m,
                                                mdp->S, mdp->T, mdp->R, mdp->V,
                                                mdp->ne, mdp->expanded,
                                                s,
                                                mdp->VPrime, mdp->pi);
        } else {
            ssp_rtdp_bellman_update_state_cpu(mdp->n, mdp->ns, mdp->m,
                                                mdp->S, mdp->T, mdp->R, mdp->VPrime,
                                                mdp->ne, mdp->expanded,
                                                s,
                                                mdp->V, mdp->pi);
        }

        // This is the greedy action.
        unsigned int a = mdp->pi[s];

        // Randomly explore the state space using the action.
        float target = (float)rand() / (float)RAND_MAX;
        float current = 0.0f;
        int sp = 0;

        for (unsigned int i = 0; i < mdp->ns; i++) {
            sp = mdp->S[s * mdp->m * mdp->ns + a * mdp->ns + i];
            if (sp < 0) {
                break;
            }

            current += mdp->T[s * mdp->m * mdp->ns + a * mdp->ns + i];
            if (current >= target) {
                break;
            }
        }

        s = sp;

        // Check if s is a goal.
        for (unsigned int i = 0; i < mdp->ng; i++) {
            if (s == mdp->goals[i]) {
                sIsGoal = true;
                break;
            }
        }
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

    // Free the memory for V, VPrime, and pi.
    if (mdp->V != nullptr) {
        delete [] mdp->V;
    }
    mdp->V = nullptr;

    if (mdp->VPrime != nullptr) {
        delete [] mdp->VPrime;
    }
    mdp->VPrime = nullptr;

    if (mdp->pi != nullptr) {
        delete [] mdp->pi;
    }
    mdp->pi = nullptr;

    return NOVA_SUCCESS;
}


int ssp_rtdp_get_policy_cpu(MDP *mdp, unsigned int &r, unsigned int *&S, float *&V, unsigned int *&pi)
{
    if (S != nullptr || V != nullptr || pi != nullptr) {
        fprintf(stderr, "Error[ssp_rtdp_get_policy_cpu]: %s\n", "Invalid arguments. S, V, and pi must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    r = mdp->ne;
    S = new unsigned int[r];
    V = new float[r];
    pi = new unsigned int[r];

    // Determine which is the source for V based on the current horizon.
    float *Vsrc = nullptr;
    if (mdp->currentHorizon % 2 == 0) {
        Vsrc = mdp->VPrime;
    } else {
        Vsrc = mdp->V;
    }

    // Copy the final (or intermediate) result, both V and pi.
    for (unsigned int i = 0; i < mdp->ne; i++) {
        unsigned int s = mdp->expanded[i];

        S[i] = s;
        V[i] = Vsrc[s];
        pi[i] = mdp->pi[s];
    }

    return NOVA_SUCCESS;
}


int ssp_rtdp_free_policy_cpu(MDP *mdp, unsigned int *&S, float *&V, unsigned int *&pi)
{
    if (S != nullptr) {
        delete [] S;
    }
    S = nullptr;

    if (V != nullptr) {
        delete [] V;
    }
    V = nullptr;

    if (pi != nullptr) {
        delete [] pi;
    }
    pi = nullptr;

    return NOVA_SUCCESS;
}

