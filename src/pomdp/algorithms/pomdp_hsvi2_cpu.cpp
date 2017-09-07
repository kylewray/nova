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


#include <nova/pomdp/algorithms/pomdp_hsvi2_cpu.h>

#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <time.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

void pomdp_hsvi2_update_compute_best_alpha_cpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, float gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    const float *Gamma, unsigned int i, unsigned int a, float *alpha)
{
    for (unsigned int o = 0; o < z; o++) {
        float value = NOVA_FLT_MIN;
        float bestValue = NOVA_FLT_MIN;
        unsigned int bestAlphaIndex = 0;

        for (unsigned int j = 0; j < r; j++) {
            // Variable 'j' represents the alpha in Gamma^{t-1}. It is this variable that we will maximize over.

            // Compute the value of this alpha-vector, by taking its dot product with belief (i.e., variable 'i').
            value = 0.0f;
            for (unsigned int k = 0; k < rz; k++) {
                int s = Z[i * rz + k];
                if (s < 0) {
                    break;
                }

                // Compute the updated value for this alpha vector from the previous time step, i.e.,
                // V^{t-1}(s, a, omega (o), alpha (j)) for all states s in S. Only do this for the non-zero
                // values, since we are just computing the value now.
                float Vtk = 0.0f;
                for (unsigned int l = 0; l < ns; l++) {
                    int sp = S[s * m * ns + a * ns + l];
                    if (sp < 0) {
                        break;
                    }

                    Vtk += O[a * n * z + sp * z + o] * T[s * m * ns + a * ns + l] * Gamma[j * n + sp];
                }
                Vtk *= gamma;

                value += Vtk * B[i * rz + k];
            }

            // Assign this as the best alpha-vector if it is better.
            if (value > bestValue) {
                bestAlphaIndex = j;
                bestValue = value;
            }
        }

        // Now that we have the best 'j' (alpha-vector), we can compute the value over *all* states and add
        // this to the current 'alpha' variable we are computing. (This step is the final step of summing
        // over beliefs of the argmax of Vt.)
        for (unsigned int s = 0; s < n; s++) {
            float Vtk = 0.0f;
            for (unsigned int l = 0; l < ns; l++) {
                int sp = S[s * m * ns + a * ns + l];
                if (sp < 0) {
                    break;
                }

                Vtk += O[a * n * z + sp * z + o] * T[s * m * ns + a * ns + l] * Gamma[bestAlphaIndex * n + sp];
            }
            Vtk *= gamma;

            alpha[s] += Vtk;
        }
    }
}


void pomdp_hsvi2_update_step_cpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, float gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    const float *Gamma, float *GammaPrime, unsigned int *piPrime)
{
    float *alpha = new float[n];

    // For each belief point, we will find the best alpha vector for this point.
    for (unsigned int i = 0; i < r; i++) {
        // Variable 'i' refers to the current belief we are examining.

        float value = NOVA_FLT_MIN;
        float bestValue = NOVA_FLT_MIN;

        // Compute the argmax alpha-vector over Gamma_B. Since Gamma_B is created from the
        // m actions, we iterate over a in A.
        for (unsigned int a = 0; a < m; a++) {
            // Variable 'a' also refers to the alpha-vector being considered in the argmax for
            // the 'best' alpha-vector, as well as the action.

            // We create alpha which initially is the reward, and we will add the optimal
            // alpha-vector for each observation in the function below.
            for (unsigned int s = 0; s < n; s++) {
                alpha[s] = R[s * m + a];
            }

            // First, compute the argmax_{alpha in Gamma_{a,omega}} for each observation.
            pomdp_hsvi2_update_compute_best_alpha_cpu(n, ns, m, z, r, rz, gamma,
                                                        S, T, O, R, Z, B, Gamma,
                                                        i, a, alpha);

            // After the alpha-vector is computed, we must compute its value.
            value = 0.0f;
            for (unsigned int k = 0; k < rz; k++) {
                int s = Z[i * rz + k];
                if (s < 0) {
                    break;
                }

                value += alpha[s] * B[i * rz + k];
            }

            // If this is a new best value, then store the alpha-vector.
            if (value > bestValue) {
                memcpy(&GammaPrime[i * n], alpha, n * sizeof(float));
                piPrime[i] = a;
                bestValue = value;
            }
        }
    }

    delete [] alpha;
    alpha = nullptr;
}


int pomdp_hsvi2_initialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    if (pomdp == nullptr || pomdp->r == 0 || pomdp->n == 0 || hsvi2 == nullptr) {
        fprintf(stderr, "Error[pomdp_hsvi2_initialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon.
    hsvi2->currentHorizon = 0;

    // Create the variables.
    hsvi2->Gamma = new float[pomdp->r *pomdp->n];
    hsvi2->GammaPrime = new float[pomdp->r * pomdp->n];
    hsvi2->pi = new unsigned int[pomdp->r];

    // If provided, copy the data form the Gamma provided, and set the default values for pi.
    if (hsvi2->GammaInitial != nullptr) {
        memcpy(hsvi2->Gamma, hsvi2->GammaInitial, pomdp->r * pomdp->n * sizeof(float));
        memcpy(hsvi2->GammaPrime, hsvi2->GammaInitial, pomdp->r * pomdp->n * sizeof(float));
        for (unsigned int i = 0; i < pomdp->r; i++) {
            hsvi2->pi[i] = 0;
        }
    } else {
        for (unsigned int i = 0; i < pomdp->r * pomdp->n; i++) {
            hsvi2->Gamma[i] = 0.0f;
            hsvi2->GammaPrime[i] = 0.0f;
        }
        for (unsigned int i = 0; i < pomdp->r; i++) {
            hsvi2->pi[i] = 0;
        }
    }

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_execute_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, POMDPAlphaVectors *policy)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            hsvi2 == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = pomdp_hsvi2_initialize_cpu(pomdp, hsvi2);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n", "Failed to initialize CPU variables.");
        return result;
    }

    // For each of the updates, run HSVI2. Note that the currentHorizon is initialized to zero
    // above, and is updated in the update function below.
    while (hsvi2->currentHorizon < pomdp->horizon) {
        //printf("HSVI2 (CPU Version) -- Iteration %i of %i\n", pomdp->currentHorizon, pomdp->horizon);

        result = pomdp_hsvi2_update_cpu(pomdp, hsvi2);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n", "Failed to perform HSVI2 update step.");
            return result;
        }
    }

    result = pomdp_hsvi2_get_policy_cpu(pomdp, hsvi2, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n", "Failed to get the policy.");
    }

    result = pomdp_hsvi2_uninitialize_cpu(pomdp, hsvi2);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_hsvi2_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_uninitialize_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    if (pomdp == nullptr || hsvi2 == nullptr) {
        fprintf(stderr, "Error[pomdp_hsvi2_uninitialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon.
    hsvi2->currentHorizon = 0;

    // Free the memory for Gamma, GammaPrime, and pi.
    if (hsvi2->Gamma != nullptr) {
        delete [] hsvi2->Gamma;
    }
    hsvi2->Gamma = nullptr;

    if (hsvi2->GammaPrime != nullptr) {
        delete [] hsvi2->GammaPrime;
    }
    hsvi2->GammaPrime = nullptr;

    if (hsvi2->pi != nullptr) {
        delete [] hsvi2->pi;
    }
    hsvi2->pi = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_update_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            hsvi2 == nullptr || hsvi2->Gamma == nullptr || hsvi2->GammaPrime == nullptr || hsvi2->pi == nullptr) {
        fprintf(stderr, "Error[pomdp_hsvi2_update_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // We oscillate between Gamma and GammaPrime depending on the step.
    if (hsvi2->currentHorizon % 2 == 0) {
        pomdp_hsvi2_update_step_cpu(pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                    pomdp->S, pomdp->T, pomdp->O, pomdp->R, pomdp->Z, pomdp->B,
                    hsvi2->Gamma, hsvi2->GammaPrime, hsvi2->pi);
    } else {
        pomdp_hsvi2_update_step_cpu(pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                    pomdp->S, pomdp->T, pomdp->O, pomdp->R, pomdp->Z, pomdp->B,
                    hsvi2->GammaPrime, hsvi2->Gamma, hsvi2->pi);
    }

    hsvi2->currentHorizon++;

    return NOVA_SUCCESS;
}


int pomdp_hsvi2_get_policy_cpu(const POMDP *pomdp, POMDPHSVI2CPU *hsvi2, POMDPAlphaVectors *policy)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->m == 0 || pomdp->r == 0 ||
            hsvi2 == nullptr || (hsvi2->currentHorizon % 2 == 0 && hsvi2->Gamma == nullptr) ||
            (hsvi2->currentHorizon % 2 == 1 && hsvi2->GammaPrime == nullptr) || hsvi2->pi == nullptr ||
            policy == nullptr) {
        fprintf(stderr, "Error[pomdp_hsvi2_get_policy_cpu]: %s\n",
                        "Invalid arguments. Policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Initialize the policy, which allocates memory.
    int result = pomdp_alpha_vectors_initialize(policy, pomdp->n, pomdp->m, pomdp->r);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_hsvi2_get_policy_cpu]: %s\n", "Could not create the policy.");
        return NOVA_ERROR_POLICY_CREATION;
    }

    // Copy the final (or intermediate) result of Gamma and pi to the variables.
    if (hsvi2->currentHorizon % 2 == 0) {
        memcpy(policy->Gamma, hsvi2->Gamma, pomdp->r * pomdp->n * sizeof(float));
    } else {
        memcpy(policy->Gamma, hsvi2->GammaPrime, pomdp->r * pomdp->n * sizeof(float));
    }
    memcpy(policy->pi, hsvi2->pi, pomdp->r * sizeof(unsigned int));

    return NOVA_SUCCESS;
}

}; // namespace nova


