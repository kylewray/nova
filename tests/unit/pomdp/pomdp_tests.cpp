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


#include "unit/pomdp/pomdp_tests.h"

#include "pomdp.h"
#include "error_codes.h"
#include "constants.h"


namespace nova {
namespace tests {

nova::POMDP *create_simple_pomdp()
{
    nova::POMDP *pomdp = new nova::POMDP();

    pomdp->n = 1;
    pomdp->ns = 1;
    pomdp->m = 1;
    pomdp->z = 1;
    pomdp->r = 1;
    pomdp->rz = 1;

    pomdp->S = new int[1];
    pomdp->S[0] = 0;

    pomdp->T = new float[1];
    pomdp->T[0] = 1.0f;

    pomdp->O = new float[1];
    pomdp->O[0] = 1.0f;

    pomdp->R = new float[1];
    pomdp->R[0] = 1.0f;

    pomdp->Z = new int[1];
    pomdp->Z[0] = 0;

    pomdp->B = new float[1];
    pomdp->B[0] = 1.0f;

    pomdp->gamma = 0.9f;
    pomdp->horizon = 3;

    return pomdp;
}

nova::POMDP *create_two_state_pomdp()
{
    nova::POMDP *pomdp = new nova::POMDP();

    pomdp->n = 2;
    pomdp->ns = 2;
    pomdp->m = 2;
    pomdp->z = 2;
    pomdp->r = 1;
    pomdp->rz = 1;

    pomdp->S = new int[pomdp->n * pomdp->m * pomdp->ns];
    pomdp->T = new float[pomdp->n * pomdp->m * pomdp->ns];
    pomdp->O = new float[pomdp->m * pomdp->n * pomdp->z];
    pomdp->R = new float[pomdp->m * pomdp->n];

    pomdp->S[0 * pomdp->m * pomdp->ns + 0 * pomdp->ns + 0] = 0;
    pomdp->T[0 * pomdp->m * pomdp->ns + 0 * pomdp->ns + 0] = 0.8f;
    pomdp->S[0 * pomdp->m * pomdp->ns + 0 * pomdp->ns + 1] = 1;
    pomdp->T[0 * pomdp->m * pomdp->ns + 0 * pomdp->ns + 1] = 0.2f;

    pomdp->S[0 * pomdp->m * pomdp->ns + 1 * pomdp->ns + 0] = 0;
    pomdp->T[0 * pomdp->m * pomdp->ns + 1 * pomdp->ns + 0] = 0.2f;
    pomdp->S[0 * pomdp->m * pomdp->ns + 1 * pomdp->ns + 1] = 1;
    pomdp->T[0 * pomdp->m * pomdp->ns + 1 * pomdp->ns + 1] = 0.8f;

    pomdp->S[1 * pomdp->m * pomdp->ns + 0 * pomdp->ns + 0] = 0;
    pomdp->T[1 * pomdp->m * pomdp->ns + 0 * pomdp->ns + 0] = 0.8f;
    pomdp->S[1 * pomdp->m * pomdp->ns + 0 * pomdp->ns + 1] = 1;
    pomdp->T[1 * pomdp->m * pomdp->ns + 0 * pomdp->ns + 1] = 0.2f;

    pomdp->S[1 * pomdp->m * pomdp->ns + 1 * pomdp->ns + 0] = 0;
    pomdp->T[1 * pomdp->m * pomdp->ns + 1 * pomdp->ns + 0] = 0.2f;
    pomdp->S[1 * pomdp->m * pomdp->ns + 1 * pomdp->ns + 1] = 1;
    pomdp->T[1 * pomdp->m * pomdp->ns + 1 * pomdp->ns + 1] = 0.8f;

    pomdp->O[0 * pomdp->n * pomdp->z + 0 * pomdp->z + 0] = 0.8f;
    pomdp->O[0 * pomdp->n * pomdp->z + 0 * pomdp->z + 1] = 0.2f;
    pomdp->O[0 * pomdp->n * pomdp->z + 1 * pomdp->z + 0] = 0.2f;
    pomdp->O[0 * pomdp->n * pomdp->z + 1 * pomdp->z + 1] = 0.8f;
    pomdp->O[1 * pomdp->n * pomdp->z + 0 * pomdp->z + 0] = 0.8f;
    pomdp->O[1 * pomdp->n * pomdp->z + 0 * pomdp->z + 1] = 0.2f;
    pomdp->O[1 * pomdp->n * pomdp->z + 1 * pomdp->z + 0] = 0.2f;
    pomdp->O[1 * pomdp->n * pomdp->z + 1 * pomdp->z + 1] = 0.8f;

    // If you are in a state, and take the action to stay in that state,
    // then there is a penalty. Otherwise, you get a bonus. Due to the
    // uncertainty regarding the success/failure of moving, and the noisy
    // observations, this makes it an interesting problem to hopefully
    // oscillate between the two states.
    pomdp->R[0 * pomdp->m + 0] = -1.0f;
    pomdp->R[0 * pomdp->m + 1] = 1.0f;
    pomdp->R[1 * pomdp->m + 0] = 1.0f;
    pomdp->R[1 * pomdp->m + 1] = -1.0f;

    // Initially, we know we are in state 0.
    pomdp->Z = new int[pomdp->r * pomdp->rz];
    pomdp->B = new float[pomdp->r * pomdp->rz];

    pomdp->Z[0 * pomdp->rz + 0] = 0;
    pomdp->B[0 * pomdp->rz + 0] = 1.0f;

    pomdp->gamma = 0.9f;
    pomdp->horizon = 3;

    return pomdp;
}

}; // namespace tests
}; // namespace nova

