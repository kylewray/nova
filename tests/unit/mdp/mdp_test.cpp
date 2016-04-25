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


#include "unit/mdp/mdp_test.h"

#include "mdp.h"
#include "error_codes.h"
#include "constants.h"


namespace nova {
namespace tests {

nova::MDP *create_simple_mdp(bool ssp)
{
    nova::MDP *mdp = new nova::MDP();

    mdp->n = 1;
    mdp->ns = 1;
    mdp->m = 1;

    mdp->S = new int[1];
    mdp->S[0] = 0;

    mdp->T = new float[1];
    mdp->T[0] = 1.0f;

    mdp->R = new float[1];
    mdp->R[0] = 1.0f;

    mdp->gamma = 0.9f;
    mdp->horizon = 3;
    mdp->epsilon = 0.0001f;

    mdp->ng = 1;
    mdp->goals = new unsigned int[1];
    mdp->goals[0] = 0;

    return mdp;
}


nova::MDP *create_three_state_mdp(bool ssp)
{
    nova::MDP *mdp = new nova::MDP();

    // States: 0 = start, 1 = intermediate, 2 = goal
    mdp->n = 3;
    mdp->ns = 2;

    // Actions: 0 = wait, 1 = move
    mdp->m = 2;

    mdp->S = new int[mdp->n * mdp->m * mdp->ns];
    mdp->T = new float[mdp->n * mdp->m * mdp->ns];
    mdp->R = new float[mdp->n * mdp->m];

    // The start-wait pair transitions to intermediate with equal probability and low cost.
    mdp->S[0 * mdp->m * mdp->ns + 0 * mdp->ns + 0] = 0;
    mdp->T[0 * mdp->m * mdp->ns + 0 * mdp->ns + 0] = 0.5f;
    mdp->S[0 * mdp->m * mdp->ns + 0 * mdp->ns + 1] = 1;
    mdp->T[0 * mdp->m * mdp->ns + 0 * mdp->ns + 1] = 0.5f;
    if (ssp) {
        mdp->R[0 * mdp->m + 0] = 1.0f;
    } else {
        mdp->R[0 * mdp->m + 0] = -1.0f;
    }

    // The start-move pair transitions to intermediate with perfect probability and high cost.
    mdp->S[0 * mdp->m * mdp->ns + 1 * mdp->ns + 0] = 0;
    mdp->T[0 * mdp->m * mdp->ns + 1 * mdp->ns + 0] = 1.0f;
    mdp->S[0 * mdp->m * mdp->ns + 1 * mdp->ns + 1] = -1;
    mdp->T[0 * mdp->m * mdp->ns + 1 * mdp->ns + 1] = 0.0f;
    if (ssp) {
        mdp->R[0 * mdp->m + 1] = 2.0f;
    } else {
        mdp->R[0 * mdp->m + 1] = -2.0f;
    }

    // The intermediate-wait pair transitions to goal with zero probability and low cost.
    mdp->S[1 * mdp->m * mdp->ns + 0 * mdp->ns + 0] = 1;
    mdp->T[1 * mdp->m * mdp->ns + 0 * mdp->ns + 0] = 1.0f;
    mdp->S[1 * mdp->m * mdp->ns + 0 * mdp->ns + 1] = -1;
    mdp->T[1 * mdp->m * mdp->ns + 0 * mdp->ns + 1] = 0.0f;
    if (ssp) {
        mdp->R[1 * mdp->m + 0] = 1.0f;
    } else {
        mdp->R[1 * mdp->m + 0] = -1.0f;
    }

    // The intermediate-move pair transitions to goal with perfect probability and high cost.
    mdp->S[1 * mdp->m * mdp->ns + 1 * mdp->ns + 0] = 2;
    mdp->T[1 * mdp->m * mdp->ns + 1 * mdp->ns + 0] = 1.0f;
    mdp->S[1 * mdp->m * mdp->ns + 1 * mdp->ns + 1] = -1;
    mdp->T[1 * mdp->m * mdp->ns + 1 * mdp->ns + 1] = 0.0f;
    if (ssp) {
        mdp->R[1 * mdp->m + 1] = 2.0f;
    } else {
        mdp->R[1 * mdp->m + 1] = -2.0f;
    }

    // The goal-wait pair transitions to goal with equal probability and zero cost.
    mdp->S[2 * mdp->m * mdp->ns + 0 * mdp->ns + 0] = 2;
    mdp->T[2 * mdp->m * mdp->ns + 0 * mdp->ns + 0] = 1.0f;
    mdp->S[2 * mdp->m * mdp->ns + 0 * mdp->ns + 1] = -1;
    mdp->T[2 * mdp->m * mdp->ns + 0 * mdp->ns + 1] = 0.0f;
    if (ssp) {
        mdp->R[2 * mdp->m + 0] = 0.0f;
    } else {
        mdp->R[2 * mdp->m + 0] = 0.0f;
    }

    // The goal-move pair transitions to goal with perfect probability and zero cost.
    mdp->S[2 * mdp->m * mdp->ns + 1 * mdp->ns + 0] = 2;
    mdp->T[2 * mdp->m * mdp->ns + 1 * mdp->ns + 0] = 1.0f;
    mdp->S[2 * mdp->m * mdp->ns + 1 * mdp->ns + 1] = -1;
    mdp->T[2 * mdp->m * mdp->ns + 1 * mdp->ns + 1] = 0.0f;
    if (ssp) {
        mdp->R[2 * mdp->m + 1] = 0.0f;
    } else {
        mdp->R[2 * mdp->m + 1] = 0.0f;
    }

    mdp->gamma = 0.9f;
    mdp->horizon = 5;
    mdp->epsilon = 0.0001f;

    mdp->ng = 1;
    mdp->goals = new unsigned int[1];
    mdp->goals[0] = 2;

    return mdp;
}

}; // namespace tests
}; // namespace nova

