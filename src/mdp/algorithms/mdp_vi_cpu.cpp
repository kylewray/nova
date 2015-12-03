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


void mdp_bellman_update_cpu(unsigned int n, unsigned int ns, unsigned int m, float gamma, 
                        const int *S, const float *T, const float *R, const float *V,
                        float *VPrime, unsigned int *pi)
{
    // The intermediate Q(s, a) value.
    float Qsa;

    // The index within S and T (i.e., in n*s*ns).
    int sIndex;

    // The true successor state index (in 0 to n-1), resolved using S.
    int spIndex;

    for (unsigned int s = 0; s < n; s++) {
        VPrime[s] = -FLT_MAX;

        // Compute max_{a in A} Q(s, a).
        for (int a = 0; a < m; a++) {
            // Compute Q(s, a) for this action.
            Qsa = R[s * m + a];

            for (int sp = 0; sp < ns; sp++) {
                sIndex = s * m * ns + a * ns + sp;

                spIndex = S[sIndex];
                if (spIndex < 0) {
                    break;
                }

                Qsa += gamma * T[sIndex] * V[spIndex];
            }

            if (a == 0 || Qsa > VPrime[s]) {
                VPrime[s] = Qsa;
                pi[s] = a;
            }
        }
    }
}


int mdp_vi_complete_cpu(MDP *mdp, const float *Vinitial, float *&V, unsigned int *&pi)
{
    // Note: This 'wrapper' function is provided in order to maintain 
    // the same structure as the GPU version. In the GPU version,
    // 'complete' performs the initilization and uninitialization of
    // the MDP object on the device as well. Here, we do not need that.
    return mdp_vi_execute_cpu(mdp, Vinitial, V, pi);
}


int mdp_vi_initialize_cpu(MDP *mdp, const float *Vinitial)
{
    // Reset the current horizon.
    mdp->currentHorizon = 0;

    // Create the variables.
    mdp->V = new float[mdp->n];
    mdp->VPrime = new float[mdp->n];
    mdp->pi = new unsigned int[mdp->n];

    // Copy the data from the V provided, and set default values for pi.
    memcpy(mdp->V, Vinitial, mdp->n * sizeof(float));
    memcpy(mdp->VPrime, Vinitial, mdp->n * sizeof(float));
    for (unsigned int i = 0; i < mdp->n; i++) {
        mdp->pi[i] = 0;
    }

    return NOVA_SUCCESS;
}


int mdp_vi_execute_cpu(MDP *mdp, const float *Vinitial, float *&V, unsigned int *&pi)
{
    int result;

    // First, ensure data is valid.
    if (mdp->n == 0 || mdp->ns == 0 || mdp->m == 0 ||
            mdp->S == nullptr || mdp->T == nullptr || mdp->R == nullptr ||
            mdp->gamma < 0.0f || mdp->gamma > 1.0f || mdp->horizon < 1 ||
            Vinitial == nullptr || V != nullptr || pi != nullptr) {
        fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    result = mdp_vi_initialize_cpu(mdp, Vinitial);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Failed to initialize the CPU variables.");
        return result;
    }

    // We iterate over all time steps up to the horizon. Initialize set the currentHorizon to 0,
    // and the update increments it.
    while (mdp->currentHorizon < mdp->horizon) {
        result = mdp_vi_update_cpu(mdp);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Failed to perform Bellman update on the CPU.");
            return result;
        }
    }

    result = mdp_vi_get_policy_cpu(mdp, V, pi);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Failed to get the policy.");
        return result;
    }

    result = mdp_vi_uninitialize_cpu(mdp);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[mdp_vi_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int mdp_vi_uninitialize_cpu(MDP *mdp)
{
    // Reset the current horizon.
    mdp->currentHorizon = 0;

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


int mdp_vi_update_cpu(MDP *mdp)
{
    // We oscillate between V and VPrime depending on the step.
    if (mdp->currentHorizon % 2 == 0) {
        mdp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->S, mdp->T, mdp->R, mdp->V, mdp->VPrime, mdp->pi);
    } else {
        mdp_bellman_update_cpu(mdp->n, mdp->ns, mdp->m, mdp->gamma, mdp->S, mdp->T, mdp->R, mdp->VPrime, mdp->V, mdp->pi);
    }

    mdp->currentHorizon++;

    return NOVA_SUCCESS;
}


int mdp_vi_get_policy_cpu(MDP *mdp, float *&V, unsigned int *&pi)
{
    if (V != nullptr || pi != nullptr) {
        fprintf(stderr, "Error[mdp_vi_get_policy_cpu]: %s\n", "Invalid arguments. V and pi must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    V = new float[mdp->n];
    pi = new unsigned int[mdp->n];

    // Copy the final (or intermediate) result, both V and pi. This assumes memory has been allocated
    // for the variables provided.
    if (mdp->currentHorizon % 2 == 0) {
        memcpy(V, mdp->V, mdp->n * sizeof(float));
    } else {
        memcpy(V, mdp->VPrime, mdp->n * sizeof(float));
    }
    memcpy(pi, mdp->pi, mdp->n * sizeof(float));

    return NOVA_SUCCESS;
}


int mdp_vi_free_policy_cpu(MDP *mdp, float *&V, unsigned int *&pi)
{
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


