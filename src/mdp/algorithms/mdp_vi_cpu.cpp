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


#include "algorithms/mdp_vi_cpu.h"
#include "error_codes.h"
#include "constants.h"

#include <stdio.h>
#include <cstring>

namespace nova {

void mdp_vi_bellman_update_cpu(unsigned int n, unsigned int ns, unsigned int m, float gamma,
        const int *S, const float *T, const float *R, const float *V,
        float *Vprime, unsigned int *pi)
{
    for (unsigned int s = 0; s < n; s++) {
        Vprime[s] = -FLT_MAX;

        // Compute max_{a in A} Q(s, a).
        for (int a = 0; a < m; a++) {
            // Compute Q(s, a) for this action.
            float Qsa = R[s * m + a];

            for (int i = 0; i < ns; i++) {
                int sp = S[s * m * ns + a * ns + i];
                if (sp < 0) {
                    break;
                }

                Qsa += gamma * T[s * m * ns + a * ns + i] * V[sp];
            }

            if (a == 0 || Qsa > Vprime[s]) {
                Vprime[s] = Qsa;
                pi[s] = a;
            }
        }
    }
}


int mdp_vi_initialize_cpu(const MDP *mdp, MDPVICPU *vi)
{
    // Reset the current horizon.
    vi->currentHorizon = 0;

    // Create the variables.
    vi->V = new float[mdp->n];
    vi->Vprime = new float[mdp->n];
    vi->pi = new unsigned int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    memcpy(vi->V, vi->Vinitial, mdp->n * sizeof(float));
    memcpy(vi->Vprime, vi->Vinitial, mdp->n * sizeof(float));
    for (unsigned int i = 0; i < mdp->n; i++) {
        vi->pi[i] = 0;
    }

    return NOVA_SUCCESS;
}


int mdp_vi_execute_cpu(const MDP *mdp, MDPVICPU *vi, MDPValueFunction *&policy)
{
    int result;

    // First, ensure data is valid.
    if (mdp == nullptr || mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->gamma < 0.0f || mdp->gamma > 1.0f || mdp->horizon < 1 ||
            vi == nullptr || vi->Vinitial == nullptr || policy != nullptr) {
        fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    result = mdp_vi_initialize_cpu(mdp, vi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Failed to initialize the CPU variables.");
        return result;
    }

    // We iterate over all time steps up to the horizon. Initialize set the currentHorizon to 0,
    // and the update increments it.
    while (vi->currentHorizon < mdp->horizon) {
        result = mdp_vi_update_cpu(mdp, vi);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Failed to perform Bellman update on the CPU.");
            return result;
        }
    }

    result = mdp_vi_get_policy_cpu(mdp, vi, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Failed to get the policy.");
        return result;
    }

    result = mdp_vi_uninitialize_cpu(mdp, vi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int mdp_vi_uninitialize_cpu(const MDP *mdp, MDPVICPU *vi)
{
    // Reset the current horizon.
    vi->currentHorizon = 0;

    // Free the memory for V, Vprime, and pi.
    if (vi->V != nullptr) {
        delete [] vi->V;
    }
    vi->V = nullptr;

    if (vi->Vprime != nullptr) {
        delete [] vi->Vprime;
    }
    vi->Vprime = nullptr;

    if (vi->pi != nullptr) {
        delete [] vi->pi;
    }
    vi->pi = nullptr;

    return NOVA_SUCCESS;
}


int mdp_vi_update_cpu(const MDP *mdp, MDPVICPU *vi)
{
    if (vi->currentHorizon % 2 == 0) {
        mdp_vi_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->S, mdp->T, mdp->R, vi->V, vi->Vprime, vi->pi);
    } else {
        mdp_vi_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->S, mdp->T, mdp->R, vi->Vprime, vi->V, vi->pi);
    }

    vi->currentHorizon++;

    return NOVA_SUCCESS;
}


int mdp_vi_get_policy_cpu(const MDP *mdp, MDPVICPU *vi, MDPValueFunction *&policy)
{
    if (policy != nullptr) {
        fprintf(stderr, "Error[mdp_vi_get_policy_cpu]: %s\n",
                        "Invalid arguments. The policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    policy = new MDPValueFunction();

    policy->n = mdp->n;
    policy->m = mdp->m;
    policy->r = 0;

    policy->S = nullptr;
    policy->V = new float[mdp->n];
    policy->pi = new unsigned int[mdp->n];

    // Copy the final (or intermediate) result, both V and pi. This assumes memory has been allocated
    // for the variables provided.
    if (vi->currentHorizon % 2 == 0) {
        memcpy(policy->V, vi->V, mdp->n * sizeof(float));
    } else {
        memcpy(policy->V, vi->Vprime, mdp->n * sizeof(float));
    }
    memcpy(policy->pi, vi->pi, mdp->n * sizeof(float));

    return NOVA_SUCCESS;
}

}; // namespace nova

