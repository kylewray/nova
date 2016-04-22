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


#include "algorithms/mdp_vi_gpu.h"

#include <gtest/gtest.h>

#include "utilities/mdp_model_cpu.h"
#include "utilities/mdp_model_gpu.h"
#include "error_codes.h"
#include "constants.h"

#include "unit/mdp/mdp_test.h"


namespace nova {
namespace tests {

TEST(MDPVIGPU, initialization)
{
    nova::MDP mdp;
    mdp.n = 2;

    nova::MDPVIGPU vi;
    vi.VInitial = nullptr;
    vi.numThreads = 512;

    int result = nova::mdp_vi_initialize_gpu(&mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_EQ(vi.VInitial, nullptr);
    EXPECT_EQ(vi.currentHorizon, 0);
    EXPECT_EQ(vi.numThreads, 512);

    EXPECT_NE(vi.d_V, nullptr);
    EXPECT_NE(vi.d_VPrime, nullptr);
    EXPECT_NE(vi.d_pi, nullptr);

    if (result == NOVA_SUCCESS) {
        result = nova::mdp_vi_uninitialize_gpu(&mdp, &vi);
        EXPECT_EQ(result, NOVA_SUCCESS);
    }

    vi.VInitial = new float[2];
    vi.VInitial[0] = -1.0f;
    vi.VInitial[1] = 1.0f;

    vi.numThreads = 512;

    result = nova::mdp_vi_initialize_gpu(&mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_NE(vi.VInitial, nullptr);
    EXPECT_EQ(vi.VInitial[0], -1.0f);
    EXPECT_EQ(vi.VInitial[1], 1.0f);

    EXPECT_EQ(vi.currentHorizon, 0);
    EXPECT_EQ(vi.numThreads, 512);

    EXPECT_NE(vi.d_V, nullptr);
    EXPECT_NE(vi.d_VPrime, nullptr);
    EXPECT_NE(vi.d_pi, nullptr);

    if (result == NOVA_SUCCESS) {
        result = nova::mdp_vi_uninitialize_gpu(&mdp, &vi);
        EXPECT_EQ(result, NOVA_SUCCESS);
    }

    delete [] vi.VInitial;
}


TEST(MDPVIGPU, badInitializations)
{
    nova::MDP mdp;
    mdp.n = 0;

    nova::MDPVIGPU vi;
    vi.VInitial = new float[2];
    vi.VInitial[0] = -1.0f;
    vi.VInitial[1] = 1.0f;
    vi.numThreads = 512;

    int result = 0;

    bool requiresCleanup = false;

    result = nova::mdp_vi_initialize_gpu(nullptr, &vi);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    if (result != NOVA_ERROR_INVALID_DATA) {
        requiresCleanup = true;
    }

    result = nova::mdp_vi_initialize_gpu(&mdp, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    if (result != NOVA_ERROR_INVALID_DATA) {
        requiresCleanup = true;
    }

    result = nova::mdp_vi_initialize_gpu(&mdp, &vi);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    if (result != NOVA_ERROR_INVALID_DATA) {
        requiresCleanup = true;
    }

    if (requiresCleanup) {
        result = nova::mdp_vi_uninitialize_gpu(&mdp, &vi);
        EXPECT_EQ(result, NOVA_SUCCESS);
    }

    delete [] vi.VInitial;
}


TEST(MDPVIGPU, execution)
{
    nova::MDP *mdp = create_simple_mdp();

    int result = nova::mdp_initialize_gpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    nova::MDPVIGPU vi;
    vi.VInitial = nullptr;
    vi.numThreads = 512;

    nova::MDPValueFunction *policy = nullptr;

    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_NE(policy, nullptr);

    if (policy != nullptr) {
        EXPECT_EQ(policy->n, 1);
        EXPECT_EQ(policy->m, 1);

        EXPECT_EQ(policy->r, 0);
        EXPECT_EQ(policy->S, nullptr);

        EXPECT_NE(policy->V, nullptr);
        if (policy->V != nullptr) {
            EXPECT_NEAR(policy->V[0], 2.71f, 1e-5);
        }

        EXPECT_NE(policy->pi, nullptr);
        if (policy->pi != nullptr) {
            EXPECT_EQ(policy->pi[0], 0);
        }
    }

    if (policy != nullptr) {
        result = nova::mdp_value_function_uninitialize(policy);
        EXPECT_EQ(result, NOVA_SUCCESS);
        delete policy;
        policy = nullptr;
    }

    result = nova::mdp_uninitialize_gpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}


TEST(MDPVIGPU, badExecution)
{
    nova::MDP *mdp = create_simple_mdp();

    int result = nova::mdp_initialize_gpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    nova::MDPVIGPU vi;
    vi.VInitial = nullptr;
    vi.numThreads = 512;

    nova::MDPValueFunction *policy = nullptr;

    result = nova::mdp_vi_execute_gpu(nullptr, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    result = nova::mdp_vi_execute_gpu(mdp, nullptr, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    policy = new nova::MDPValueFunction();
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    mdp->n = 0;
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->n = 1;

    mdp->ns = 0;
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->ns = 1;

    mdp->m = 0;
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->m = 1;

    void *tempAddress = nullptr;

    tempAddress = mdp->S;
    mdp->S = nullptr;
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->S = (int *)tempAddress;

    tempAddress = mdp->T;
    mdp->T = nullptr;
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->T = (float *)tempAddress;

    tempAddress = mdp->R;
    mdp->R = nullptr;
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->R = (float *)tempAddress;

    mdp->gamma = -1.0f;
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->gamma = 0.9f;

    mdp->gamma = 2.0f;
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->gamma = 0.9f;

    mdp->horizon = 0;
    result = nova::mdp_vi_execute_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->horizon = 3;

    if (policy != nullptr) {
        result = nova::mdp_value_function_uninitialize(policy);
        EXPECT_EQ(result, NOVA_SUCCESS);
        delete policy;
        policy = nullptr;
    }

    result = nova::mdp_uninitialize_gpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}


TEST(MDPVIGPU, uninitialization)
{
    nova::MDP mdp;
    mdp.n = 1;

    nova::MDPVIGPU vi;
    vi.VInitial = nullptr;
    vi.numThreads = 512;

    int result = nova::mdp_vi_initialize_gpu(&mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);

    result = nova::mdp_vi_uninitialize_gpu(&mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_EQ(vi.currentHorizon, 0);
    EXPECT_EQ(vi.d_V, nullptr);
    EXPECT_EQ(vi.d_VPrime, nullptr);
    EXPECT_EQ(vi.d_pi, nullptr);
}


TEST(MDPVIGPU, badUninitialization)
{
    nova::MDP mdp;
    mdp.n = 1;

    int result = nova::mdp_vi_uninitialize_gpu(nullptr, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    result = nova::mdp_vi_uninitialize_gpu(&mdp, nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
}


// For performance reasons, there is no checking here, so we simply
// are checking valid mathematics.
TEST(MDPVIGPU, update)
{
    nova::MDP *mdp = create_simple_mdp();

    int result = nova::mdp_initialize_gpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    nova::MDPVIGPU vi;
    vi.VInitial = nullptr;
    vi.numThreads = 512;

    nova::MDPValueFunction *policy = nullptr;

    result = nova::mdp_vi_initialize_gpu(mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);

    vi.currentHorizon = 0;

    result = nova::mdp_vi_update_gpu(mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);
    result = nova::mdp_vi_get_policy_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_SUCCESS);
    EXPECT_EQ(vi.currentHorizon, 1);
    EXPECT_EQ(policy->V[0], 1.0f);
    EXPECT_EQ(policy->pi[0], 0);
    result = nova::mdp_value_function_uninitialize(policy);
    EXPECT_EQ(result, NOVA_SUCCESS);
    delete policy;
    policy = nullptr;

    result = nova::mdp_vi_update_gpu(mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);
    result = nova::mdp_vi_get_policy_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_SUCCESS);
    EXPECT_EQ(vi.currentHorizon, 2);
    EXPECT_EQ(policy->V[0], 1.9f);
    EXPECT_EQ(policy->pi[0], 0);
    result = nova::mdp_value_function_uninitialize(policy);
    EXPECT_EQ(result, NOVA_SUCCESS);
    delete policy;
    policy = nullptr;

    result = nova::mdp_vi_update_gpu(mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);
    result = nova::mdp_vi_get_policy_gpu(mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_SUCCESS);
    EXPECT_EQ(vi.currentHorizon, 3);
    EXPECT_EQ(policy->V[0], 2.71f);
    EXPECT_EQ(policy->pi[0], 0);
    result = nova::mdp_value_function_uninitialize(policy);
    EXPECT_EQ(result, NOVA_SUCCESS);
    delete policy;
    policy = nullptr;

    result = nova::mdp_vi_uninitialize_gpu(mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);

    result = nova::mdp_uninitialize_gpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}


// There is no bad update because for performance reasons, we do not
// check or sanitize input to this function.
//TEST(MDPVIGPU, badUpdate) { }


TEST(MDPVIGPU, getPolicy)
{
    nova::MDP mdp;
    mdp.n = 2;
    mdp.m = 3;

    nova::MDPVIGPU vi;
    vi.VInitial = new float[2];
    vi.VInitial[0] = 10.0f;
    vi.VInitial[1] = 20.0f;
    vi.numThreads = 512;

    int result = nova::mdp_vi_initialize_gpu(&mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);

    nova::MDPValueFunction *policy = nullptr;

    result = nova::mdp_vi_get_policy_gpu(&mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_SUCCESS);

    EXPECT_NE(policy, nullptr);

    if (policy != nullptr) {
        EXPECT_EQ(policy->n, mdp.n);
        EXPECT_EQ(policy->m, mdp.m);

        EXPECT_EQ(policy->r, 0);
        EXPECT_EQ(policy->S, nullptr);

        EXPECT_NE(policy->V, nullptr);
        EXPECT_EQ(policy->V[0], 10.0f);
        EXPECT_EQ(policy->V[1], 20.0f);

        EXPECT_NE(policy->V, nullptr);
        EXPECT_EQ(policy->pi[0], 0);
        EXPECT_EQ(policy->pi[1], 0);

        result = nova::mdp_value_function_uninitialize(policy);
        EXPECT_EQ(result, NOVA_SUCCESS);
        delete policy;
        policy = nullptr;
    }

    result = nova::mdp_vi_uninitialize_gpu(&mdp, &vi);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete [] vi.VInitial;
}


TEST(MDPVIGPU, badGetPolicy)
{
    nova::MDP mdp;

    nova::MDPVIGPU vi;
    vi.VInitial = nullptr;
    vi.numThreads = 512;

    nova::MDPValueFunction *policy = new nova::MDPValueFunction();

    int result = nova::mdp_vi_get_policy_gpu(nullptr, nullptr, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    result = nova::mdp_vi_get_policy_gpu(&mdp, nullptr, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    result = nova::mdp_vi_get_policy_gpu(&mdp, &vi, policy);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    delete policy;
}

}; // namespace tests
}; // namespace nova


