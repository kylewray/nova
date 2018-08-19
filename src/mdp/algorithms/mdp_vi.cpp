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


#include <nova/mdp/algorithms/mdp_vi.h>

#include <stdio.h>
#include <cstring>

#include <nova/mdp/policies/mdp_value_function.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

void mdp_vi_bellman_update(unsigned int n, unsigned int ns, unsigned int m, float gamma,
        const int *S, const float *T, const float *R, const float *V,
        float *VPrime, unsigned int *pi)
{
    for (unsigned int s = 0; s < n; s++) {
        VPrime[s] = -NOVA_FLT_MAX;

        // Compute max_{a in A} Q(s, a).
        for (unsigned int a = 0; a < m; a++) {
            // Compute Q(s, a) for this action.
            float Qsa = R[s * m + a];

            for (unsigned int i = 0; i < ns; i++) {
                int sp = S[s * m * ns + a * ns + i];
                if (sp < 0) {
                    break;
                }

                Qsa += gamma * T[s * m * ns + a * ns + i] * V[sp];
            }

            if (a == 0 || Qsa > VPrime[s]) {
                VPrime[s] = Qsa;
                pi[s] = a;
            }
        }
    }
}


int mdp_vi_execute(const MDP *mdp, MDPVI *vi, MDPValueFunction *policy)
{
    // First, ensure data is valid.
    if (mdp == nullptr || mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->gamma < 0.0f || mdp->gamma > 1.0f || mdp->horizon < 1 ||
            vi == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[mdp_vi_execute]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = mdp_vi_initialize(mdp, vi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute]: %s\n", "Failed to initialize the variables.");
        return result;
    }

    // We iterate over all time steps up to the horizon. Initialize set the currentHorizon to 0,
    // and the update increments it.
    while (vi->currentHorizon < mdp->horizon) {
        result = mdp_vi_update(mdp, vi);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[mdp_vi_execute]: %s\n", "Failed to perform Bellman update on the .");

            int resultPrime = mdp_vi_uninitialize(mdp, vi);
            if (resultPrime != NOVA_SUCCESS) {
                fprintf(stderr, "Error[mdp_vi_execute]: %s\n", "Failed to uninitialize the  variables.");
            }
            return result;
        }
    }

    result = mdp_vi_get_policy(mdp, vi, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute]: %s\n", "Failed to get the policy.");
    }

    result = mdp_vi_uninitialize(mdp, vi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute]: %s\n", "Failed to uninitialize the variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int mdp_vi_initialize(const MDP *mdp, MDPVI *vi)
{
    if (mdp == nullptr || mdp->n == 0 || vi == nullptr) {
        fprintf(stderr, "Error[mdp_vi_initialize]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon.
    vi->currentHorizon = 0;

    // Create the variables.
    vi->V = new float[mdp->n];
    vi->VPrime = new float[mdp->n];
    vi->pi = new unsigned int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    // If undefined, then assign 0 for V.
    if (vi->VInitial != nullptr) {
        memcpy(vi->V, vi->VInitial, mdp->n * sizeof(float));
        memcpy(vi->VPrime, vi->VInitial, mdp->n * sizeof(float));
        for (unsigned int i = 0; i < mdp->n; i++) {
            vi->pi[i] = 0;
        }
    } else {
        for (unsigned int i = 0; i < mdp->n; i++) {
            vi->V[i] = 0.0f;
            vi->VPrime[i] = 0.0f;
            vi->pi[i] = 0;
        }
    }

    return NOVA_SUCCESS;
}


int mdp_vi_update(const MDP *mdp, MDPVI *vi)
{
    if (vi->currentHorizon % 2 == 0) {
        mdp_vi_bellman_update(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->S, mdp->T, mdp->R, vi->V, vi->VPrime, vi->pi);
    } else {
        mdp_vi_bellman_update(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->S, mdp->T, mdp->R, vi->VPrime, vi->V, vi->pi);
    }

    vi->currentHorizon++;

    return NOVA_SUCCESS;
}


int mdp_vi_get_policy(const MDP *mdp, MDPVI *vi, MDPValueFunction *policy)
{
    if (mdp == nullptr || vi == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[mdp_vi_get_policy]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Initialize the policy, which allocates memory.
    int result = mdp_value_function_initialize(policy, mdp->n, mdp->m, 0);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_get_policy]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy the final (or intermediate) result, both V and pi. This assumes memory has been allocated
    // for the variables provided.
    if (vi->currentHorizon % 2 == 0) {
        memcpy(policy->V, vi->V, mdp->n * sizeof(float));
    } else {
        memcpy(policy->V, vi->VPrime, mdp->n * sizeof(float));
    }
    memcpy(policy->pi, vi->pi, mdp->n * sizeof(float));

    return NOVA_SUCCESS;
}


int mdp_vi_uninitialize(const MDP *mdp, MDPVI *vi)
{
    if (mdp == nullptr || vi == nullptr) {
        fprintf(stderr, "Error[mdp_vi_uninitialize]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon.
    vi->currentHorizon = 0;

    // Free the memory for V, VPrime, and pi.
    if (vi->V != nullptr) {
        delete [] vi->V;
    }
    vi->V = nullptr;

    if (vi->VPrime != nullptr) {
        delete [] vi->VPrime;
    }
    vi->VPrime = nullptr;

    if (vi->pi != nullptr) {
        delete [] vi->pi;
    }
    vi->pi = nullptr;

    return NOVA_SUCCESS;
}

}; // namespace nova

