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


#include <nova/pomdp/algorithms/pomdp_perseus_cpu.h>

#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <time.h>

#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

void pomdp_perseus_compute_b_dot_alpha_cpu(unsigned int rz, const int *Z, const float *B, unsigned int bIndex,
    const float *alpha, float *bDotAlpha)
{
    *bDotAlpha = 0.0f;

    for (unsigned int j = 0; j < rz; j++) {
        int s = Z[bIndex * rz + j];
        if (s < 0) {
            break;
        }

        *bDotAlpha += B[bIndex * rz + j] * alpha[s];
    }
}


void pomdp_perseus_compute_Vb_cpu(unsigned int n, unsigned int rz, const int *Z, const float *B, unsigned int bIndex,
    const float *Gamma, unsigned int rGamma, float *Vnb, unsigned int *alphaPrimeIndex)
{
    *Vnb = NOVA_FLT_MIN;

    for (unsigned int i = 0; i < rGamma; i++) {
        float bDotAlpha = NOVA_FLT_MIN;

        pomdp_perseus_compute_b_dot_alpha_cpu(rz, Z, B, bIndex, &Gamma[i * n], &bDotAlpha);

        if (*Vnb < bDotAlpha) {
            *Vnb = bDotAlpha;
            *alphaPrimeIndex = i;
        }
    }
}


void pomdp_perseus_update_compute_best_alpha_cpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, float gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    unsigned int bIndex, const float *Gamma, unsigned int rGamma, unsigned int a, float *alpha)
{
    for (unsigned int o = 0; o < z; o++) {
        float value = NOVA_FLT_MIN;
        float bestValue = NOVA_FLT_MIN;
        unsigned int bestAlphaIndex = 0;

        for (unsigned int j = 0; j < rGamma; j++) {
            // Variable 'j' represents the alpha in Gamma^{t-1}. It is this variable that we will maximize over.

            // Compute the value of this alpha-vector, by taking its dot product with belief (i.e., variable 'i').
            value = 0.0f;
            for (unsigned int k = 0; k < rz; k++) {
                int s = Z[bIndex * rz + k];
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

                value += Vtk * B[bIndex * rz + k];
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


void pomdp_perseus_update_step_cpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, float gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    const float *Gamma, unsigned int rGamma,
    unsigned int bIndex, float *alphaPrime, unsigned int *aPrime)
{
    float value = NOVA_FLT_MIN;
    float bestValue = NOVA_FLT_MIN;

    float *alpha = new float[n];

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
        pomdp_perseus_update_compute_best_alpha_cpu(n, ns, m, z, r, rz, gamma,
                                                    S, T, O, R, Z, B, bIndex,
                                                    Gamma, rGamma, a, alpha);

        // After the alpha-vector is computed, we must compute its value.
        pomdp_perseus_compute_b_dot_alpha_cpu(rz, Z, B, bIndex, alpha, &value);

        // If this is a new best value, then store the alpha-vector.
        if (value > bestValue) {
            memcpy(alphaPrime, alpha, n * sizeof(float));
            *aPrime = a;
            bestValue = value;
        }
    }

    delete [] alpha;
}


int pomdp_perseus_initialize_cpu(const POMDP *pomdp, POMDPPerseusCPU *per)
{
    if (pomdp == nullptr || pomdp->r == 0 || pomdp->n == 0 || per == nullptr) {
        fprintf(stderr, "Error[pomdp_perseus_initialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon.
    per->currentHorizon = 0;

    // Create the variables.
    per->Gamma = new float[pomdp->r * pomdp->n];
    per->GammaPrime = new float[pomdp->r * pomdp->n];
    per->pi = new unsigned int[pomdp->r];
    per->piPrime = new unsigned int[pomdp->r];

    // If provided, copy the data form the Gamma provided, and set the default values for pi.
    if (per->GammaInitial != nullptr) {
        memcpy(per->Gamma, per->GammaInitial, pomdp->r * pomdp->n * sizeof(float));
        memcpy(per->GammaPrime, per->GammaInitial, pomdp->r * pomdp->n * sizeof(float));
        for (unsigned int i = 0; i < pomdp->r; i++) {
            per->pi[i] = 0;
            per->piPrime[i] = 0;
        }
    } else {
        for (unsigned int i = 0; i < pomdp->r * pomdp->n; i++) {
            per->Gamma[i] = 0.0f;
            per->GammaPrime[i] = 0.0f;
        }
        for (unsigned int i = 0; i < pomdp->r; i++) {
            per->pi[i] = 0;
            per->piPrime[i] = 0;
        }
    }

    // For Perseus, we might have a lot fewer alpha-vectors. The actual number is
    // given by rGamma and rGammaPrime. Initially, the set V_n and V_n' are empty,
    // which is equivalent to setting 0.
    per->rGamma = 0;
    per->rGammaPrime = 0;

    // Finally, we have BTilde, which stores the indexes of the relevant belief points
    // that require updating at each step. Convergence occurs when this set is empty.
    // Initially, BTilde = B.
    per->rTilde = pomdp->r;
    per->BTilde = new unsigned int[pomdp->r];

    for (unsigned int i = 0; i < pomdp->r; i++) {
        per->BTilde[i] = i;
    }

    return NOVA_SUCCESS;
}


int pomdp_perseus_execute_cpu(const POMDP *pomdp, POMDPPerseusCPU *per, POMDPAlphaVectors *policy)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            per == nullptr || policy == nullptr) {
        fprintf(stderr, "Error[pomdp_perseus_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    int result = pomdp_perseus_initialize_cpu(pomdp, per);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_perseus_execute_cpu]: %s\n", "Failed to initialize CPU variables.");
        return result;
    }

    // For each of the updates, run Perseus. The update function checks for convergence and will terminate
    // the loop early (if BTilde is empty). Also, note that the currentHorizon is initialized to zero above,
    // and is updated in the update function below.
    while (per->currentHorizon < pomdp->horizon) {
        //printf("Perseus (CPU Version) -- Iteration %i of %i\n", pomdp->currentHorizon, pomdp->horizon);

        // Note: "Convergence" here means iterating until Btilde is empty, which is always guaranteed to
        // be at most r iterations (the number of belief points).
        result = NOVA_SUCCESS;

        while (result != NOVA_CONVERGED) {
            result = pomdp_perseus_update_cpu(pomdp, per);
            if (result != NOVA_CONVERGED && result != NOVA_SUCCESS) {
                fprintf(stderr, "Error[pomdp_perseus_execute_cpu]: %s\n", "Failed to perform Perseus update step.");
                return result;
            }
        }
    }

    result = pomdp_perseus_get_policy_cpu(pomdp, per, policy);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_perseus_execute_cpu]: %s\n", "Failed to get the policy.");
    }

    result = pomdp_perseus_uninitialize_cpu(pomdp, per);
    if (result != NOVA_SUCCESS) {
        fprintf(stderr, "Error[pomdp_perseus_execute_cpu]: %s\n", "Failed to uninitialize the CPU variables.");
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_perseus_uninitialize_cpu(const POMDP *pomdp, POMDPPerseusCPU *per)
{
    if (pomdp == nullptr || per == nullptr) {
        fprintf(stderr, "Error[pomdp_perseus_uninitialize_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Reset the current horizon.
    per->currentHorizon = 0;

    // Free the memory for Gamma, GammaPrime, and pi.
    if (per->Gamma != nullptr) {
        delete [] per->Gamma;
    }
    per->Gamma = nullptr;
    per->rGamma = 0;

    if (per->GammaPrime != nullptr) {
        delete [] per->GammaPrime;
    }
    per->GammaPrime = nullptr;
    per->rGammaPrime = 0;

    if (per->pi != nullptr) {
        delete [] per->pi;
    }
    per->pi = nullptr;

    if (per->piPrime != nullptr) {
        delete [] per->piPrime;
    }
    per->piPrime = nullptr;

    // Free the memory of BTilde and reset rTilde.
    if (per->BTilde != nullptr) {
        delete [] per->BTilde;
    }
    per->BTilde = nullptr;
    per->rTilde = 0;

    return NOVA_SUCCESS;
}


int pomdp_perseus_update_cpu(const POMDP *pomdp, POMDPPerseusCPU *per)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 ||
            pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0f || pomdp->gamma > 1.0f || pomdp->horizon < 1 ||
            per == nullptr || per->rTilde > pomdp->r || per->BTilde == nullptr ||
            per->Gamma == nullptr || per->GammaPrime == nullptr ||
            per->pi == nullptr || per->piPrime == nullptr) {
        fprintf(stderr, "Error[pomdp_perseus_update_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // For convenience, define a variable pointing to the proper Gamma and rGamma variables.
    // Note: Gamma == V_n and GammPrime == V_{n+1} from the paper.
    float *Gamma = nullptr;
    float *GammaPrime = nullptr;
    unsigned int *rGamma = nullptr;
    unsigned int *rGammaPrime = nullptr;
    unsigned int *pi = nullptr;
    unsigned int *piPrime = nullptr;

    if (per->currentHorizon % 2 == 0) {
        Gamma = per->Gamma;
        GammaPrime = per->GammaPrime;
        rGamma = &per->rGamma;
        rGammaPrime = &per->rGammaPrime;
        pi = per->pi;
        piPrime = per->piPrime;
    } else {
        Gamma = per->GammaPrime;
        GammaPrime = per->Gamma;
        rGamma = &per->rGammaPrime;
        rGammaPrime = &per->rGamma;
        pi = per->piPrime;
        piPrime = per->pi;
    }

    // Sample a belief point at random from BTilde.
    unsigned int bTildeIndex = (unsigned int)((float)rand() / (float)RAND_MAX * (float)(per->rTilde));
    unsigned int bIndex = per->BTilde[bTildeIndex];

    // Perform one Bellman update to compute the optimal alpha-vector and action for this belief point (bIndex).
    float *alpha = new float[pomdp->n];
    unsigned int alphaAction = 0;

    pomdp_perseus_update_step_cpu(pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                pomdp->S, pomdp->T, pomdp->O, pomdp->R, pomdp->Z, pomdp->B,
                Gamma, *rGamma,
                bIndex, alpha, &alphaAction);

    // First compute the value of this *new* alpha-vector at this belief.
    float bDotAlpha = 0.0f;

    pomdp_perseus_compute_b_dot_alpha_cpu(pomdp->rz, pomdp->Z, pomdp->B, bIndex, alpha, &bDotAlpha);

    // Next, for each alpha-vector, we will compute the alpha-dot-b for this belief (bIndex),
    // using the *old* alpha-vectors. This also recalls which alpha-vector obtained this value
    // (for the else case below).
    float Vnb = 0.0f;
    unsigned int alphaPrimeIndex = 0;

    pomdp_perseus_compute_Vb_cpu(pomdp->n, pomdp->rz, pomdp->Z, pomdp->B, bIndex, Gamma, *rGamma, &Vnb, &alphaPrimeIndex);

    // Now, if this new alpha-vector improved the value at bIndex, then add it to the set of alpha-vectors.
    // Otherwise, add the best alpha-vector from the current set of alpha-vectors.
    if (bDotAlpha >= Vnb) {
        memcpy(&GammaPrime[(*rGammaPrime) * pomdp->n], alpha, pomdp->n * sizeof(float));
        piPrime[*rGammaPrime] = alphaAction;
    } else {
        memcpy(&GammaPrime[(*rGammaPrime) * pomdp->n], &Gamma[alphaPrimeIndex * pomdp->n], pomdp->n * sizeof(float));
        piPrime[*rGammaPrime] = pi[alphaPrimeIndex];
    }
    (*rGammaPrime)++;

    delete [] alpha;
    alpha = nullptr;

    if (*rGammaPrime > pomdp->r) {
        fprintf(stderr, "Error[pomdp_perseus_update_cpu]: %s\n", "Out of memory. Too many alpha-vectors added.");
        return NOVA_ERROR_OUT_OF_MEMORY;
    }

    // Compute BTilde again, which consists of all beliefs that degraded in value after adding whatever
    // alpha-vector was added in the if-else statement above. Trivially, we are guaranteed to have removed
    // belief bIndex, and this set strictly shrinks in size. Ideally, many more beliefs had values which
    // improved, so it should shrink quite rapidly, especially for the first few iterations.
    per->rTilde = 0;

    for (unsigned int i = 0; i < pomdp->r; i++) {
        unsigned int action = 0;

        Vnb = 0.0f;
        float Vnp1b = 0.0f;

        pomdp_perseus_compute_Vb_cpu(pomdp->n, pomdp->rz,  pomdp->Z, pomdp->B, i, Gamma, *rGamma, &Vnb, &action);
        pomdp_perseus_compute_Vb_cpu(pomdp->n, pomdp->rz,  pomdp->Z, pomdp->B, i, GammaPrime, *rGammaPrime, &Vnp1b, &action);

        if (Vnp1b < Vnb) {
            per->BTilde[per->rTilde] = i;
            per->rTilde++;
        }
    }

    // Check for convergence (if BTilde is empty).
    if (per->rTilde == 0) {
        // We performed one complete step of Perseus for this horizon!
        per->currentHorizon++;

        // Note #1: The way this code is written puts the reset here. In the original paper, it is
        // at the beginning of the next horizon's set of iterations.

        // Note #2: Resetting rGamma here still allows for the correct set of policies to be grabbed via
        // the get_policy function below, because the horizon was incremented, so the GammaPrime values
        // will actually be retrieved in get_policy.

        // Reset V_{n+1} to empty set.
        *rGamma = 0;

        // Reset BTilde to B.
        per->rTilde = pomdp->r;
        for (unsigned int i = 0; i < pomdp->r; i++) {
            per->BTilde[i] = i;
        }

        return NOVA_CONVERGED;
    }

    return NOVA_SUCCESS;
}


int pomdp_perseus_get_policy_cpu(const POMDP *pomdp, POMDPPerseusCPU *per, POMDPAlphaVectors *policy)
{
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->m == 0 ||
            per == nullptr || (per->currentHorizon % 2 == 0 &&
                (per->rGamma == 0 || per->Gamma == nullptr  || per->pi == nullptr)) ||
            (per->currentHorizon % 2 == 1 &&
                (per->rGammaPrime == 0 || per->GammaPrime == nullptr || per->piPrime == nullptr)) ||
            policy == nullptr) {
        fprintf(stderr, "Error[pomdp_perseus_get_policy_cpu]: %s\n",
                        "Invalid arguments. Policy must be undefined.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Copy the final (or intermediate) result of Gamma and pi to the variables.
    if (per->currentHorizon % 2 == 0) {
        int result = pomdp_alpha_vectors_initialize(policy, pomdp->n, pomdp->m, per->rGamma);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[pomdp_perseus_get_policy_cpu]: %s\n", "Could not create the policy.");
            return NOVA_ERROR_POLICY_CREATION;
        }

        memcpy(policy->Gamma, per->Gamma, per->rGamma * pomdp->n * sizeof(float));
        memcpy(policy->pi, per->pi, per->rGamma * sizeof(unsigned int));
    } else {
        int result = pomdp_alpha_vectors_initialize(policy, pomdp->n, pomdp->m, per->rGammaPrime);
        if (result != NOVA_SUCCESS) {
            fprintf(stderr, "Error[pomdp_perseus_get_policy_cpu]: %s\n", "Could not create the policy.");
            return NOVA_ERROR_POLICY_CREATION;
        }

        memcpy(policy->Gamma, per->GammaPrime, per->rGammaPrime * pomdp->n * sizeof(float));
        memcpy(policy->pi, per->piPrime, per->rGammaPrime * sizeof(unsigned int));
    }

    return NOVA_SUCCESS;
}

}; // namespace nova

