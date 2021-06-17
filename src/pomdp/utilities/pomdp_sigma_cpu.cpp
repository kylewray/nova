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


#include <nova/pomdp/utilities/pomdp_sigma_cpu.h>

#include <stdio.h>
#include <algorithm>
#include <vector>

#include <nova/error_codes.h>
#include <nova/constants.h>

namespace nova {

// A quick typedef for comparing beliefs and remembering their indexes.
typedef std::pair<float, int> SigmaPair;


/**
 *  A comparator function for SigmaPair types. It returns true if left is greater
 *  than right, false otherwise.
 */
bool pomdp_sigma_pair_comparator_cpu(const SigmaPair &bl, const SigmaPair &br)
{
    return bl.first > br.first;
}


int pomdp_sigma_cpu(POMDP *pomdp, unsigned int numDesiredNonZeroValues, float &sigma)
{
    // Ensure the data is valid.
    if (pomdp == nullptr || pomdp->z == 0 || pomdp->r == 0 || pomdp->rz == 0 ||
            pomdp->Z == nullptr || pomdp->B == nullptr ||
            numDesiredNonZeroValues == 0) {
        fprintf(stderr, "Error[pomdp_sigma_cpu]: %s\n", "Invalid arguments.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Create the belief return variables.
    int *Znew = new int[pomdp->r * numDesiredNonZeroValues];
    float *Bnew = new float[pomdp->r * numDesiredNonZeroValues];
    for (unsigned int i = 0; i < numDesiredNonZeroValues * pomdp->n; i++) {
        Znew[i] = -1;
        Bnew[i] = 0.0f;
    }

    sigma = 1.0;

    // For each belief point, we need to sort the values in descending order, then take
    // the top numDesiredNonZeroValues belief values, normalize them, and store them in
    // the correct positions in Bnew.
    for (unsigned int i = 0; i < pomdp->r; i++) {
        // Construct the belief point and sort, remembering the original indexes.
        std::vector<SigmaPair> b;
        for (unsigned int k = 0; k < pomdp->rz; k++) {
            b.push_back(SigmaPair(pomdp->B[i * pomdp->rz + k], pomdp->Z[i * pomdp->rz + k]));
        }

        std::sort(b.begin(), b.end(), pomdp_sigma_pair_comparator_cpu);

        // Compute the normalization constant (sigma_b).
        float sigmab = 0.0f;
        for (unsigned int k = 0; k < numDesiredNonZeroValues; k++) {
            sigmab += b[k].first;
        }

        if (sigmab < sigma) {
            sigma = sigmab;
        }

        // Take the top rz belief values to construct the new Bnew.
        for (unsigned int k = 0; k < numDesiredNonZeroValues; k++) {
            if (b[k].first > 0.0f) {
                Bnew[i * numDesiredNonZeroValues + k] = b[k].first / sigmab;
                Znew[i * numDesiredNonZeroValues + k] = b[k].second;
            } else {
                Bnew[i * numDesiredNonZeroValues + k] = 0.0f;
                Znew[i * numDesiredNonZeroValues + k] = -1;
            }
        }
    }

    // Free the current belief variables.
    if (pomdp->Z != nullptr) {
        delete [] pomdp->Z;
    }
    if (pomdp->B != nullptr) {
        delete [] pomdp->B;
    }

    // Assign the new values for belief variables.
    pomdp->rz = numDesiredNonZeroValues;
    pomdp->Z = Znew;
    pomdp->B = Bnew;

    return NOVA_SUCCESS;
}

}; // namespace nova

