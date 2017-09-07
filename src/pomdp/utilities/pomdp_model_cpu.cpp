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

#include <nova/pomdp/utilities/pomdp_model_cpu.h>

#include <stdio.h>
#include <cmath>

#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

int pomdp_initialize_cpu(POMDP *pomdp, unsigned int n, unsigned int ns, unsigned int m,
    unsigned int z, unsigned int r, unsigned int rz, float gamma, unsigned int horizon)
{
    if (pomdp == nullptr || pomdp->S != nullptr || pomdp->T != nullptr || pomdp->O != nullptr ||
            pomdp->R != nullptr || pomdp->Z != nullptr || pomdp->B != nullptr ||
            n == 0 || ns == 0 || n < ns || m == 0 || z == 0 || r == 0 || rz == 0 ||
            gamma < 0.0f || gamma > 1.0f || horizon == 0) {
        fprintf(stderr, "Error[pomdp_initialize_cpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    pomdp->n = n;
    pomdp->ns = ns;
    pomdp->m = m;
    pomdp->z = z;
    pomdp->r = r;
    pomdp->rz = rz;
    pomdp->gamma = gamma;
    pomdp->horizon = horizon;

    pomdp->S = new int[pomdp->n * pomdp->m * pomdp->ns];
    pomdp->T = new float[pomdp->n * pomdp->m * pomdp->ns];
    for (unsigned int i = 0; i < pomdp->n * pomdp->m * pomdp->ns; i++) {
        pomdp->S[i] = -1;
        pomdp->T[i] = 0.0f;
    }

    pomdp->O = new float[pomdp->m * pomdp->n * pomdp->z];
    for (unsigned int i = 0; i < pomdp->m * pomdp->n * pomdp->z; i++) {
        pomdp->O[i] = 0.0f;
    }

    pomdp->R = new float[pomdp->n * pomdp->m];
    for (unsigned int i = 0; i < pomdp->n * pomdp->m; i++) {
        pomdp->R[i] = 0.0f;
    }

    pomdp->Z = new int[pomdp->r * pomdp->rz];
    pomdp->B = new float[pomdp->r * pomdp->rz];
    for (unsigned int i = 0; i < pomdp->r * pomdp->rz; i++) {
        pomdp->Z[i] = 0.0f;
        pomdp->B[i] = 0.0f;
    }

    pomdp->d_S = nullptr;
    pomdp->d_T = nullptr;
    pomdp->d_O = nullptr;
    pomdp->d_R = nullptr;
    pomdp->d_Z = nullptr;
    pomdp->d_B = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_belief_update_cpu(const POMDP *pomdp, const float *b, unsigned int a, unsigned int o, float *&bp)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 || pomdp->z == 0 ||
            pomdp->S == nullptr || pomdp->T == nullptr || pomdp->O == nullptr || pomdp->R == nullptr ||
            bp != nullptr) {
        fprintf(stderr, "Error[pomdp_belief_update_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Create the resultant belief vector and assign default values.
    bp = new float[pomdp->n];
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

    // If the normalizing constant is exceedingly small, within error tolerances, then this is
    // very likely to be an invalid belief. In practice, this arises when there is a probabilistically
    // impossible observation, given the POMDP.
    if (std::fabs(normalizingConstant) < NOVA_FLT_ERR_TOL) {
        fprintf(stderr, "Error[pomdp_belief_update_cpu]: %s\n",
                "Computed belief is invalid. The observation is impossible given the belief and action.");
        return NOVA_WARNING_INVALID_BELIEF;
    }

    for (unsigned int sp = 0; sp < pomdp->n; sp++) {
        bp[sp] /= normalizingConstant;
    }

    return NOVA_SUCCESS;
}


int pomdp_expand_update_max_non_zero_values_cpu(const POMDP *pomdp, const float *b,
    unsigned int &maxNonZeroValues)
{
    unsigned int numNonZeroValues = 0;
    for (unsigned int s = 0; s < pomdp->n; s++) {
        if (b[s] > 0.0f) {
            numNonZeroValues++;
        }
    }
    if (numNonZeroValues > maxNonZeroValues) {
        maxNonZeroValues = numNonZeroValues;
    }

    return NOVA_SUCCESS;
}


int pomdp_add_new_raw_beliefs_cpu(POMDP *pomdp, unsigned int numBeliefsToAdd, float *Bnew)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->n == 0 || pomdp->ns == 0 || pomdp->m == 0 || pomdp->z == 0 ||
            numBeliefsToAdd == 0 || Bnew == nullptr) {
        fprintf(stderr, "Error[pomdp_add_new_beliefs_raw_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    unsigned int rFinal = pomdp->r + numBeliefsToAdd;
    unsigned int rzFinal = pomdp->rz;

    // Compute the new maximimum number of non-zero values.
    for (unsigned int i = 0; i < numBeliefsToAdd; i++) {
        unsigned int maxNonZeroValues = 0;
        pomdp_expand_update_max_non_zero_values_cpu(pomdp, &Bnew[i * pomdp->n + 0], maxNonZeroValues);

        if (rzFinal < maxNonZeroValues) {
            rzFinal = maxNonZeroValues;
        }
    }

    int *Zfinal = new int[rFinal * rzFinal];
    float *Bfinal = new float[rFinal * rzFinal];

    // Copy the original Z and B.
    for (unsigned int i = 0; i < pomdp->r; i++) {
        for (unsigned int j = 0; j < pomdp->rz; j++) {
            Zfinal[i * rzFinal + j] = pomdp->Z[i * pomdp->rz + j];
            Bfinal[i * rzFinal + j] = pomdp->B[i * pomdp->rz + j];
        }

        for (unsigned int j = pomdp->rz; j < rzFinal; j++) {
            Zfinal[i * rzFinal + j] = -1;
            Bfinal[i * rzFinal + j] = 0.0f;
        }
    }

    // Copy the new Z and B.
    for (unsigned int i = pomdp->r; i < pomdp->r + numBeliefsToAdd; i++) {
        unsigned int j = 0;

        for (unsigned int s = 0; s < pomdp->n; s++) {
            if (Bnew[(i - pomdp->r) * pomdp->n + s] > 0.0f) {
                Zfinal[i * rzFinal + j] = s;
                Bfinal[i * rzFinal + j] = Bnew[(i - pomdp->r) * pomdp->n + s];
                j++;
            }
        }

        // The remaining values are set to -1 and 0.0f.
        while (j < rzFinal) {
            Zfinal[i * rzFinal + j] = -1;
            Bfinal[i * rzFinal + j] = 0.0f;
            j++;
        }
    }

    // Free the old data and add the new data.
    if (pomdp->Z != nullptr) {
        delete [] pomdp->Z;
    }
    if (pomdp->B != nullptr) {
        delete [] pomdp->B;
    }

    pomdp->r = rFinal;
    pomdp->rz = rzFinal;
    pomdp->Z = Zfinal;
    pomdp->B = Bfinal;

    return NOVA_SUCCESS;
}


int pomdp_uninitialize_cpu(POMDP *pomdp)
{
    if (pomdp == nullptr) {
        fprintf(stderr, "Error[pomdp_uninitialize_cpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    pomdp->n = 0;
    pomdp->ns = 0;
    pomdp->m = 0;
    pomdp->z = 0;
    pomdp->r = 0;
    pomdp->rz = 0;
    pomdp->gamma = 0.0f;
    pomdp->horizon = 0;

    if (pomdp->S != nullptr) {
        delete [] pomdp->S;
    }
    pomdp->S = nullptr;

    if (pomdp->T != nullptr) {
        delete [] pomdp->T;
    }
    pomdp->T = nullptr;

    if (pomdp->O != nullptr) {
        delete [] pomdp->O;
    }
    pomdp->O = nullptr;

    if (pomdp->R != nullptr) {
        delete [] pomdp->R;
    }
    pomdp->R = nullptr;

    if (pomdp->Z != nullptr) {
        delete [] pomdp->Z;
    }
    pomdp->Z = nullptr;

    if (pomdp->B != nullptr) {
        delete [] pomdp->B;
    }
    pomdp->B = nullptr;

    return NOVA_SUCCESS;
}

}; // namespace nova

