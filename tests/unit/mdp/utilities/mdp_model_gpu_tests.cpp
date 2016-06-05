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


#include <nova/mdp/utilities/mdp_model_gpu.h>

#include <gtest/gtest.h>

#include <nova/mdp/utilities/mdp_model_cpu.h>
#include <nova/error_codes.h>
#include <nova/constants.h>

#include <unit/mdp/mdp_tests.h>

namespace nova {
namespace tests {

TEST(MDPModelGPU, initializationAndUninitialization)
{
    nova::MDP *mdp = create_simple_mdp(true);

    int result = nova::mdp_initialize_gpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    if (mdp->ng > 0) {
        EXPECT_NE(mdp->d_goals, nullptr);
    } else {
        EXPECT_EQ(mdp->d_goals, nullptr);
    }
    EXPECT_NE(mdp->d_S, nullptr);
    EXPECT_NE(mdp->d_T, nullptr);
    EXPECT_NE(mdp->d_R, nullptr);

    result = nova::mdp_uninitialize_gpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}

TEST(MDPModelGPU, badInitializations)
{
    int result = nova::mdp_initialize_gpu(nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);

    nova::MDP *mdp = create_simple_mdp(true);

    unsigned int tempValue = 0;
    void *tempAddress = nullptr;

    tempValue = mdp->n;
    mdp->n = 0;
    result = nova::mdp_initialize_successors_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    result = nova::mdp_initialize_state_transitions_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    result = nova::mdp_initialize_rewards_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->n = tempValue;

    tempValue = mdp->m;
    mdp->m = 0;
    result = nova::mdp_initialize_successors_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    result = nova::mdp_initialize_state_transitions_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    result = nova::mdp_initialize_rewards_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->m = tempValue;

    tempValue = mdp->ns;
    mdp->ns = 0;
    result = nova::mdp_initialize_successors_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    result = nova::mdp_initialize_state_transitions_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->ns = tempValue;

    tempValue = mdp->ng;
    mdp->ng = 0;
    result = nova::mdp_initialize_goals_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->ng = tempValue;

    tempAddress = mdp->S;
    mdp->S = nullptr;
    result = nova::mdp_initialize_successors_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->S = (int *)tempAddress;

    tempAddress = mdp->T;
    mdp->T = nullptr;
    result = nova::mdp_initialize_state_transitions_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->T = (float *)tempAddress;

    tempAddress = mdp->R;
    mdp->R = nullptr;
    result = nova::mdp_initialize_rewards_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->R = (float *)tempAddress;

    tempAddress = mdp->goals;
    mdp->goals = nullptr;
    result = nova::mdp_initialize_goals_gpu(mdp);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
    mdp->goals = (unsigned int *)tempAddress;

    result = nova::mdp_uninitialize_cpu(mdp);
    EXPECT_EQ(result, NOVA_SUCCESS);

    delete mdp;
}

TEST(MDPModelGPU, badUninitializations)
{
    int result = nova::mdp_uninitialize_gpu(nullptr);
    EXPECT_EQ(result, NOVA_ERROR_INVALID_DATA);
}

}; // namespace tests
}; // namespace nova



