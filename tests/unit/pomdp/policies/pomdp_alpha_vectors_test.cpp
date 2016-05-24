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


#include "policies/pomdp_alpha_vectors.h"

#include <gtest/gtest.h>

#include "error_codes.h"
#include "constants.h"


namespace nova {
namespace tests {

TEST(POMDPAlphaVectors, valueAndAction)
{
    nova::POMDPAlphaVectors pomdpAlphaVectors;
    pomdpAlphaVectors.n = 2;
    pomdpAlphaVectors.m = 2;
    pomdpAlphaVectors.r = 2;

    pomdpAlphaVectors.Gamma = new float[2 * 2];
    pomdpAlphaVectors.Gamma[0 * 2 + 0] = 1.0f;
    pomdpAlphaVectors.Gamma[0 * 2 + 1] = 0.0f;
    pomdpAlphaVectors.Gamma[1 * 2 + 0] = 0.0f;
    pomdpAlphaVectors.Gamma[1 * 2 + 1] = 1.0f;

    pomdpAlphaVectors.pi = new unsigned int[2];
    pomdpAlphaVectors.pi[0] = 1;
    pomdpAlphaVectors.pi[1] = 0;

    int result = 0;
    float b[2];
    float Vb = 0.0f;
    unsigned int a = 0;

    b[0] = 0.0f;
    b[1] = 1.0f;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_SUCCESS);
    EXPECT_NEAR(Vb, 1.0f, 1e-5f);
    EXPECT_EQ(a, 0);

    b[0] = 1.0f;
    b[1] = 0.0f;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_SUCCESS);
    EXPECT_NEAR(Vb, 1.0f, 1e-5f);
    EXPECT_EQ(a, 1);

    b[0] = 0.25f;
    b[1] = 0.75f;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_SUCCESS);
    EXPECT_NEAR(Vb, 0.75f, 1e-5f);
    EXPECT_EQ(a, 0);

    b[0] = 0.67f;
    b[1] = 0.33f;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_SUCCESS);
    EXPECT_NEAR(Vb, 0.67f, 1e-5f);
    EXPECT_EQ(a, 1);

    b[0] = 0.5f;
    b[1] = 0.5f;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_SUCCESS);
    EXPECT_NEAR(Vb, 0.5f, 1e-5f);
    //EXPECT_EQ(a, 0);      // It could be either...

    EXPECT_NE(pomdpAlphaVectors.Gamma, nullptr);
    if (pomdpAlphaVectors.Gamma != nullptr) {
        delete [] pomdpAlphaVectors.Gamma;
    }

    EXPECT_NE(pomdpAlphaVectors.pi, nullptr);
    if (pomdpAlphaVectors.pi != nullptr) {
        delete [] pomdpAlphaVectors.pi;
    }
}

TEST(POMDPAlphaVectors, badValueAndAction)
{
    nova::POMDPAlphaVectors pomdpAlphaVectors;
    pomdpAlphaVectors.n = 2;
    pomdpAlphaVectors.m = 2;
    pomdpAlphaVectors.r = 2;
    pomdpAlphaVectors.Gamma = new float[2 * 2];
    pomdpAlphaVectors.pi = new unsigned int[2];

    int result = 0;
    float b[2];
    float Vb = 0.0f;
    unsigned int a = 0;

    result = nova::pomdp_alpha_vectors_value_and_action(nullptr, b, Vb, a);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    pomdpAlphaVectors.n = 0;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    pomdpAlphaVectors.n = 2;

    pomdpAlphaVectors.m = 0;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    pomdpAlphaVectors.m = 2;

    pomdpAlphaVectors.r = 0;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    pomdpAlphaVectors.r = 2;

    void *tempAddress = nullptr;

    tempAddress = pomdpAlphaVectors.Gamma;
    pomdpAlphaVectors.Gamma = nullptr;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    pomdpAlphaVectors.Gamma = (float *)tempAddress;

    tempAddress = pomdpAlphaVectors.pi;
    pomdpAlphaVectors.pi = nullptr;
    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, b, Vb, a);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    pomdpAlphaVectors.pi = (unsigned int *)tempAddress;

    result = nova::pomdp_alpha_vectors_value_and_action(&pomdpAlphaVectors, nullptr, Vb, a);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    EXPECT_NE(pomdpAlphaVectors.Gamma, nullptr);
    if (pomdpAlphaVectors.Gamma != nullptr) {
        delete [] pomdpAlphaVectors.Gamma;
    }

    EXPECT_NE(pomdpAlphaVectors.pi, nullptr);
    if (pomdpAlphaVectors.pi != nullptr) {
        delete [] pomdpAlphaVectors.pi;
    }
}

TEST(POMDPAlphaVectors, uninitialization)
{
    nova::POMDPAlphaVectors pomdpAlphaVectors;
    pomdpAlphaVectors.n = 2;
    pomdpAlphaVectors.m = 2;
    pomdpAlphaVectors.r = 2;

    pomdpAlphaVectors.Gamma = new float[2 * 2];
    pomdpAlphaVectors.pi = new unsigned int[2];

    int result = nova::pomdp_alpha_vectors_uninitialize(&pomdpAlphaVectors);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_EQ(pomdpAlphaVectors.Gamma, nullptr);
    if (pomdpAlphaVectors.Gamma != nullptr) {
        delete [] pomdpAlphaVectors.Gamma;
    }

    EXPECT_EQ(pomdpAlphaVectors.pi, nullptr);
    if (pomdpAlphaVectors.pi != nullptr) {
        delete [] pomdpAlphaVectors.pi;
    }
}

TEST(POMDPAlphaVectors, badUninitializations)
{
    int result = nova::pomdp_alpha_vectors_uninitialize(nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
}

}; // namespace tests
}; // namespace nova


