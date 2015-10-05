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


#include "pomdp_sigma_cpu.h"
#include "error_codes.h"

#include <stdio.h>
#include <algorithm>


bool pomdp_sigma_pair_comparator_cpu(const SigmaPair &bl, const SigmaPair &br)
{
    return bl.first > br.first;
}


int pomdp_sigma_cpu(POMDP *pomdp, unsigned int rz, float *Bnew, int *Znew)
{
    // Ensure valid input.
    if (rz == 0 || Bnew == nullptr || Znew == nullptr) {
        fprintf(stderr, "Error[pomdp_sigma_cpu]: %s\n", "Invalid data.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Note: We assume Bnew has been created to be an r-rz array (the *new* rz provided).

    // For each belief point, we need to sort the values in descending order, then take
    // the top rz belief values, normalize them, and store them in the correct positions
    // in Bnew.
    for (unsigned int i = 0; i < pomdp->r; i++) {
        // Construct the belief point and sort, remembering the original indexes.
        std::vector<SigmaPair> b;
        for (unsigned int k = 0; k < pomdp->rz; k++) {
            b.push_back(SigmaPair(pomdp->B[i * pomdp->rz + k], k));
        }

        std::sort(b.begin(), b.end(), pomdp_sigma_pair_comparator_cpu);

        // Compute the normalization constant (sigma_b).
        float sigma = 0.0f;
        for (unsigned int k = 0; k < rz; k++) {
            sigma += b[k].first;
        }

        // Take the top rz belief values to construct the new Bnew.
        for (unsigned int k = 0; k < rz; k++) {
            if (b[k].first > 0.0f) {
                Bnew[i * rz + k] = b[k].first / sigma;
                Znew[i * rz + k] = b[k].second;
            } else {
                Bnew[i * rz + k] = 0.0f;
                Znew[i * rz + k] = -1;
            }
        }
    }

    return NOVA_SUCCESS;
}

