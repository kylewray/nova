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


#include "pomdp_pbvi_cpu.h"
#include "error_codes.h"

#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <time.h>


// This is determined by hardware, so what is below is a 'safe' guess. If this is off, the
// program might return 'nan' or 'inf'. These come from IEEE floating-point standards.
#define FLT_MAX 1e+35
#define FLT_MIN -1e+35
#define FLT_ERR_TOL 1e-9


void pomdp_pbvi_update_compute_best_alpha_cpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, float gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    const float *Gamma, unsigned int i, unsigned int a, float *alpha)
{
    float value;
    float bestValue;
    unsigned int bestj;

    for (unsigned int o = 0; o < z; o++) {
        value = FLT_MIN;
        bestValue = FLT_MIN;

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
                bestj = j;
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

                Vtk += O[a * n * z + sp * z + o] * T[s * m * ns + a * ns + l] * Gamma[bestj * n + sp];
            }
            Vtk *= gamma;

            alpha[s] += Vtk;
        }
    }
}


void pomdp_pbvi_update_step_cpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, float gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    const float *Gamma, float *GammaPrime, unsigned int *piPrime)
{
    float *alpha;
    float value;
    float bestValue;

    alpha = new float[n];

    // For each belief point, we will find the best alpha vector for this point.
    for (unsigned int i = 0; i < r; i++) {
        // Variable 'i' refers to the current belief we are examining.

        value = FLT_MIN;
        bestValue = FLT_MIN;

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
            pomdp_pbvi_update_compute_best_alpha_cpu(n, ns, m, z, r, rz, gamma,
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
}


int pomdp_pbvi_complete_cpu(POMDP *pomdp, float *Gamma, unsigned int *pi)
{
    // Note: This 'wrapper' function is provided in order to maintain the same structure
    // as the GPU version. In the GPU version, 'complete' performs the initialization
    // and uninitialization of the POMDP object on the device as well. Here, we do not
    // need that.
    return pomdp_pbvi_execute_cpu(pomdp, Gamma, pi);
}


int pomdp_pbvi_initialize_cpu(POMDP *pomdp, float *Gamma)
{
    // Reset the current horizon.
    pomdp->currentHorizon = 0;

    // Create the variables.
    pomdp->Gamma = new float[pomdp->r *pomdp->n];
    pomdp->GammaPrime = new float[pomdp->r * pomdp->n];
    pomdp->pi = new unsigned int[pomdp->r];

    // Copy the data form the Gamma provided, and set the default values for pi.
    memcpy(pomdp->Gamma, Gamma, pomdp->r * pomdp->n * sizeof(float));
    memcpy(pomdp->GammaPrime, Gamma, pomdp->r * pomdp->n * sizeof(float));
    for (unsigned int i = 0; i < pomdp->r; i++) {
        pomdp->pi[i] = 0;
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_execute_cpu(POMDP *pomdp, float *Gamma, unsigned int *pi)
{
    // The result from calling other functions.
    int result;

    // Ensure the data is valid.
    if (pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 || pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            pomdp->gamma < 0.0 || pomdp->gamma >= 1.0 || pomdp->horizon < 1) {
        fprintf(stderr, "Error[pomdp_pbvi_execute_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    result = pomdp_pbvi_initialize_cpu(pomdp, Gamma);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    // For each of the updates, run PBVI. Note that the currentHorizon is initialized to zero
    // above, and is updated in the update function below.
    while (pomdp->currentHorizon < pomdp->horizon) {
        //printf("PBVI (CPU Version) -- Iteration %i of %i\n", pomdp->currentHorizon, pomdp->horizon);

        result = pomdp_pbvi_update_cpu(pomdp);
        if (result != NOVA_SUCCESS) {
            return result;
        }
    }

    result = pomdp_pbvi_get_policy_cpu(pomdp, Gamma, pi);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    result = pomdp_pbvi_uninitialize_cpu(pomdp);
    if (result != NOVA_SUCCESS) {
        return result;
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_uninitialize_cpu(POMDP *pomdp)
{
    // Reset the current horizon.
    pomdp->currentHorizon = 0;

    // Free the memory for Gamma, GammaPrime, and pi.
    if (pomdp->Gamma != nullptr) {
        delete [] pomdp->Gamma;
    }
    pomdp->Gamma = nullptr;

    if (pomdp->GammaPrime != nullptr) {
        delete [] pomdp->GammaPrime;
    }
    pomdp->GammaPrime = nullptr;

    if (pomdp->pi != nullptr) {
        delete [] pomdp->pi;
    }
    pomdp->pi = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_update_cpu(POMDP *pomdp)
{
    // We oscillate between Gamma and GammaPrime depending on the step.
    if (pomdp->currentHorizon % 2 == 0) {
        pomdp_pbvi_update_step_cpu(pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                    pomdp->S, pomdp->T, pomdp->O, pomdp->R, pomdp->Z, pomdp->B,
                    pomdp->Gamma, pomdp->GammaPrime, pomdp->pi);
    } else {
        pomdp_pbvi_update_step_cpu(pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                    pomdp->S, pomdp->T, pomdp->O, pomdp->R, pomdp->Z, pomdp->B,
                    pomdp->GammaPrime, pomdp->Gamma, pomdp->pi);
    }

    pomdp->currentHorizon++;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_get_policy_cpu(POMDP *pomdp, float *Gamma, unsigned int *pi)
{
    // Copy the final (or intermediate) result of Gamma and pi to the variables. This assumes
    // that the memory has been allocated for the variables provided.
    if (pomdp->currentHorizon % 2 == 0) {
        memcpy(Gamma, pomdp->Gamma, pomdp->r * pomdp->n * sizeof(float));
    } else {
        memcpy(Gamma, pomdp->GammaPrime, pomdp->r * pomdp->n * sizeof(float));
    }
    memcpy(pi, pomdp->pi, pomdp->r * sizeof(unsigned int));

    return NOVA_SUCCESS;
}


int pomdp_pbvi_expand_belief_update_cpu(POMDP *pomdp, const float *b, unsigned int a, unsigned int o, float *bp)
{
    for (unsigned int sp = 0; sp < pomdp->n; sp++) {
        bp[sp] = 0.0f;
    }

    for (unsigned int s = 0; s < pomdp->n; s++) {
        for (unsigned int i = 0; i < pomdp->ns; i++) {
            int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + i];
            if (sp < 0) {
                break;
            }

            bp[sp] += pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + i] * b[s];
        }
    }

    float normalizingConstant = 0.0f;

    for (unsigned int sp = 0; sp < pomdp->n; sp++) {
        bp[sp] *= pomdp->O[a * pomdp->n * pomdp->z + sp * pomdp->z + o];

        normalizingConstant += bp[sp];
    }

    for (unsigned int sp = 0; sp < pomdp->n; sp++) {
        bp[sp] /= normalizingConstant;
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_expand_probability_observation(POMDP *pomdp, const float *b, unsigned int a, unsigned int o, float &prObs)
{
    prObs = 0.0f;

    for (unsigned int s = 0; s < pomdp->n; s++) {
        float val = 0.0f;

        for (unsigned int i = 0; i < pomdp->ns; i++) {
            int sp = pomdp->S[s * pomdp->m * pomdp->ns + a * pomdp->ns + i];
            if (sp < 0) {
                break;
            }

            val += pomdp->T[s * pomdp->m * pomdp->ns + a * pomdp->ns + i] *
                    pomdp->O[a * pomdp->n * pomdp->z + sp * pomdp->z + o];
        }

        prObs += val * b[s];
    }

    return NOVA_SUCCESS;
}


int pomdp_pbvi_expand_random_cpu(POMDP *pomdp, unsigned int numDesiredBeliefPoints, unsigned int *maxNonZeroValues, float *Bnew)
{
    srand(time(nullptr));

    // Setup the initial belief point.
    float *b0 = new float[pomdp->n];
    for (unsigned int s = 0; s < pomdp->n; s++) {
        b0[s] = 0.0f;
    }
    for (unsigned int i = 0; i < pomdp->rz; i++) {
        int s = pomdp->Z[0 * pomdp->rz + i];
        if (s < 0) {
            break;
        }
        b0[s] = pomdp->B[0 * pomdp->rz + i];
    }

    float *b = new float[pomdp->n];
    unsigned int i = 0;

    // For each belief point we want to expand. Each step will generate a new trajectory
    // and add the resulting belief point to B.
    while (i < numDesiredBeliefPoints) {
        // Randomly pick a horizon for this trajectory. We do this because some domains transition
        // beliefs away from areas on the (n-1)-simplex, never to return. This ensures many paths
        // are added.
        unsigned int h = (unsigned int)((float)rand() / (float)RAND_MAX * (float)(pomdp->horizon + 1));

        // Setup the belief used in exploration.
        memcpy(b, b0, pomdp->n * sizeof(float));

        // Follow a random trajectory with length equal to this horizon.
        for (unsigned int t = 0; t < h; t++) {
            // Randomly pick an action.
            unsigned int a = (unsigned int)((float)rand() / (float)RAND_MAX * (float)pomdp->m);

            float currentNumber = 0.0f;
            float targetNumber = (float)rand() / (float)RAND_MAX;

            unsigned int o = 0;
            for (unsigned int op = 0; op < pomdp->z; op++) {
                float prObs = 0.0f;
                pomdp_pbvi_expand_probability_observation(pomdp, b, a, op, prObs);
                currentNumber += prObs;

                if (currentNumber >= targetNumber) {
                    o = op;
                    break;
                }
            }

            // Follow the belief update equation to compute b' for all state primes s'.
            float *bp = new float[pomdp->n];
            pomdp_pbvi_expand_belief_update_cpu(pomdp, b, a, o, bp);
            memcpy(b, bp, pomdp->n * sizeof(float));
            delete [] bp;

            // Determine how many non-zero values exist and update rz.
            unsigned int numNonZeroValues = 0;
            for (unsigned int s = 0; s < pomdp->n; s++) {
                if (b[s] > 0.0f) {
                    numNonZeroValues++;
                }
            }
            if (numNonZeroValues > *maxNonZeroValues) {
                *maxNonZeroValues = numNonZeroValues;
            }

            // Assign the computed belief for this trajectory.
            memcpy(&Bnew[i * pomdp->n], b, pomdp->n * sizeof(float));

            // Stop if we have met the belief point quota.
            i++;
            if (i >= numDesiredBeliefPoints) {
                break;
            }
        }
    }

    delete [] b;
    delete [] b0;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_expand_ger_epsilon_cpu(POMDP *pomdp, float &epsilon)
{
    epsilon = 0.0f;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_expand_ger_cpu(POMDP *pomdp)
{

    return NOVA_SUCCESS;
}

