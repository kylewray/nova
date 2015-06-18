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


#include "mdp_model_gpu.h"
#include "error_codes.h"

#include <stdio.h>


int mdp_initialize_successors_gpu(MDP *mdp)
{
    // Ensure the data is valid.
    if (mdp->n == 0 || mdp->m == 0 || mdp->ns == 0 || mdp->S == nullptr) {
        fprintf(stderr, "Error[mdp_initialize_successors_gpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&mdp->d_S, mdp->n * mdp->m * mdp->ns * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_initialize_successors_gpu]: %s\n",
                "Failed to allocate device-side memory for the successor states.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(mdp->d_S, mdp->S, mdp->n * mdp->m * mdp->ns * sizeof(int),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_initialize_successors_gpu]: %s\n",
                "Failed to copy memory from host to device for the successor states.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int mdp_uninitialize_successors_gpu(MDP *mdp)
{
    if (mdp->d_S != nullptr) {
        if (cudaFree(mdp->d_S) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_uninitialize_successors_gpu]: %s\n",
                    "Failed to allocate device-side memory for the successor states.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    mdp->d_S = nullptr;

    return NOVA_SUCCESS;
}


int mdp_initialize_state_transitions_gpu(MDP *mdp)
{
    // Ensure the data is valid.
    if (mdp->n == 0 || mdp->m == 0 || mdp->ns == 0 || mdp->T == nullptr) {
        fprintf(stderr, "Error[mdp_initialize_state_transitions_gpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&mdp->d_T, mdp->n * mdp->m * mdp->ns * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_initialize_state_transitions_gpu]: %s\n",
                "Failed to allocate device-side memory for the state transitions.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(mdp->d_T, mdp->T, mdp->n * mdp->m * mdp->ns * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[nova_mdp_pbvi_initialize_state_transitions]: %s\n",
                "Failed to copy memory from host to device for the state transitions.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int mdp_uninitialize_state_transitions_gpu(MDP *mdp)
{
    if (mdp->d_T != nullptr) {
        if (cudaFree(mdp->d_T) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_uninitialize_state_transitions_gpu]: %s\n",
                    "Failed to allocate device-side memory for the state transitions.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    mdp->d_T = nullptr;

    return NOVA_SUCCESS;
}


int mdp_initialize_rewards_gpu(MDP *mdp)
{
    // Ensure the data is valid.
    if (mdp->n == 0 || mdp->m == 0 || mdp->R == nullptr) {
        fprintf(stderr, "Error[mdp_initialize_rewards_gpu]: %s\n", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&mdp->d_R, mdp->n * mdp->m * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_initialize_rewards_gpu]: %s\n",
                "Failed to allocate device-side memory for the rewards.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(mdp->d_R, mdp->R, mdp->n * mdp->m * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[mdp_initialize_rewards_gpu]: %s\n",
                "Failed to copy memory from host to device for the rewards.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int mdp_uninitialize_rewards_gpu(MDP *mdp)
{
    if (mdp->d_R != nullptr) {
        if (cudaFree(mdp->d_R) != cudaSuccess) {
            fprintf(stderr, "Error[mdp_uninitialize_rewards_gpu]: %s\n",
                    "Failed to allocate device-side memory for the rewards.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    mdp->d_R = nullptr;

    return NOVA_SUCCESS;
}

