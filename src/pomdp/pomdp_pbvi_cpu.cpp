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


// This is determined by hardware, so what is below is a 'safe' guess. If this is off, the
// program might return 'nan' or 'inf'. These come from IEEE floating-point standards.
#define FLT_MAX 1e+35
#define FLT_MIN -1e+35
#define FLT_ERR_TOL 1e-9

void pomdp_pbvi_update_step_cpu(unsigned int n, unsigned int ns, unsigned int m, unsigned int z,
    unsigned int r, unsigned int rz, unsigned int gamma,
    const int *S, const float *T, const float *O, const float *R, const int *Z, const float *B,
    const float *Gamma, const unsigned int *pi, float *GammaPrime, unsigned int *piPrime)
{
    
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
    pomdp->piPrime = new unsigned int[pomdp->r];

    // Copy the data form the Gamma provided, and set the default values for pi.
    memcpy(pomdp->Gamma, Gamma, pomdp->r * pomdp->rz * sizeof(float));
    memcpy(pomdp->GammaPrime, Gamma, pomdp->r * pomdp->rz * sizeof(float));
    for (unsigned int i = 0; i < pomdp->r; i++) {
        pomdp->pi[i] = 0;
        pomdp->piPrime[i] = 0;
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

    // Free the memory for Gamma, GammaPrime, pi, and piPrime.
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

    if (pomdp->piPrime != nullptr) {
        delete [] pomdp->piPrime;
    }
    pomdp->piPrime = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_update_cpu(POMDP *pomdp)
{
    // We oscillate between <Gamma, pi> and <GammaPrime, piPrime> depending on the step.
    if (pomdp->currentHorizon % 2 == 0) {
        pomdp_pbvi_update_step_cpu(pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                    pomdp->S, pomdp->T, pomdp->O, pomdp->R, pomdp->Z, pomdp->B,
                    pomdp->Gamma, pomdp->pi, pomdp->GammaPrime, pomdp->piPrime);
    } else {
        pomdp_pbvi_update_step_cpu(pomdp->n, pomdp->ns, pomdp->m, pomdp->z, pomdp->r, pomdp->rz, pomdp->gamma,
                    pomdp->S, pomdp->T, pomdp->O, pomdp->R, pomdp->Z, pomdp->B,
                    pomdp->GammaPrime, pomdp->piPrime, pomdp->Gamma, pomdp->pi);
    }

    pomdp->currentHorizon++;

    return NOVA_SUCCESS;
}


int pomdp_pbvi_get_policy_cpu(POMDP *pomdp, float *Gamma, float *pi)
{
    // Copy the final (or intermediate) result of Gamma and pi to the variables. This assumes
    // that the memory has been allocated for the variables provided.
    if (pomdp->currentHorizon % 2 == 0) {
        memcpy(Gamma, pomdp->Gamma, pomdp->r * pomdp->rz * sizeof(float));
        memcpy(pi, pomdp->pi, pomdp->r * sizeof(unsigned int));
    } else {
        memcpy(Gamma, pomdp->GammaPrime, pomdp->r * pomdp->rz * sizeof(float));
        memcpy(pi, pomdp->piPrime, pomdp->r * sizeof(unsigned int));
    }

    return NOVA_SUCCESS;
}

