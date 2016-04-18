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

#include "utilities/pomdp_model_cpu.h"

#include <stdio.h>
#include <cmath>

#include "error_codes.h"
#include "constants.h"

namespace nova {

int pomdp_belief_update_cpu(const POMDP *pomdp, const float *b, unsigned int a, unsigned int o, float *bp)
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

    // If the normalizing constant is exceedingly small, within error tolerances, then this is
    // very likely to be an invalid belief. In practice, this arises when there is a probabilistically
    // impossible observation, given the POMDP.
    if (std::fabs(normalizingConstant) < FLT_ERR_TOL) {
        return NOVA_WARNING_INVALID_BELIEF;
    }

    for (unsigned int sp = 0; sp < pomdp->n; sp++) {
        bp[sp] /= normalizingConstant;
    }

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

