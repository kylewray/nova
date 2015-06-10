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


#include "pomdp_model_gpu.h"
#include "error_codes.h"

#include <stdio.h>


int pomdp_initialize_belief_points_gpu(unsigned int n, unsigned int r, const float *B,
        float *&d_B)
{
    // Ensure the data is valid.
    if (n == 0 || r == 0 || B == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_belief_points_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_B, r * n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_belief_points_gpu]: %s",
                "Failed to allocate device-side memory for the belief points.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_B, B, r * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_belief_points_gpu]: %s",
                "Failed to copy memory from host to device for the belief points.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}

int pomdp_uninitialize_belief_points_gpu(float *&d_B)
{
    if (d_B != nullptr) {
        if (cudaFree(d_B) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_belief_points_gpu]: %s",
                    "Failed to allocate device-side memory for the belief points.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    d_B = nullptr;

    return NOVA_SUCCESS;
}

int pomdp_initialize_state_transitions_gpu(unsigned int n, unsigned int m, const float *T,
        float *&d_T)
{
    // Ensure the data is valid.
    if (n == 0 || m == 0 || T == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_state_transitions_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_T, n * m * n * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_state_transitions_gpu]: %s",
                "Failed to allocate device-side memory for the state transitions.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_T, T, n * m * n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[nova_pomdp_pbvi_initialize_state_transitions]: %s",
                "Failed to copy memory from host to device for the state transitions.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}

int pomdp_uninitialize_state_transitions_gpu(float *&d_T)
{
    if (d_T != nullptr) {
        if (cudaFree(d_T) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_state_transitions_gpu]: %s",
                    "Failed to allocate device-side memory for the state transitions.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    d_T = nullptr;

    return NOVA_SUCCESS;
}

int pomdp_initialize_observation_transitions_gpu(unsigned int n, unsigned int m, unsigned int z,
        const float *O, float *&d_O)
{
    // Ensure the data is valid.
    if (n == 0 || m == 0 || z == 0 || O == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_observation_transitions_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_O, m * n * z * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_observation_transitions_gpu]: %s",
                "Failed to allocate device-side memory for the observation transitions.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_O, O, m * n * z * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_observation_transitions_gpu]: %s",
                "Failed to copy memory from host to device for the observation transitions.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}

int pomdp_uninitialize_observation_transitions_gpu(float *&d_O)
{
    if (d_O != nullptr) {
        if (cudaFree(d_O) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_observation_transitions_gpu]: %s",
                    "Failed to allocate device-side memory for the observation transitions.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    d_O = nullptr;

    return NOVA_SUCCESS;
}

int pomdp_initialize_rewards_gpu(unsigned int n, unsigned int m, const float *R, float *&d_R)
{
    // Ensure the data is valid.
    if (n == 0 || m == 0 || R == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_rewards_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_R, n * m * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_rewards_gpu]: %s",
                "Failed to allocate device-side memory for the rewards.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_R, R, n * m * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_rewards_gpu]: %s",
                "Failed to copy memory from host to device for the rewards.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}

int pomdp_uninitialize_rewards_gpu(float *&d_R)
{
    if (d_R != nullptr) {
        if (cudaFree(d_R) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_rewards_gpu]: %s",
                    "Failed to allocate device-side memory for the rewards.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    d_R = nullptr;

    return NOVA_SUCCESS;
}

int pomdp_initialize_available_gpu(unsigned int m, unsigned int r,
        const bool *available, bool *&d_available)
{
    // Ensure the data is valid.
    if (m == 0 || r == 0 || available == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_available_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_available, r * m * sizeof(bool)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_available_gpu]: %s",
                "Failed to allocate device-side memory for the available actions.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_available, available, r * m * sizeof(bool),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_available_gpu]: %s",
                "Failed to copy memory from host to device for the available actions.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}

int pomdp_uninitialize_available_gpu(bool *&d_available)
{
    if (d_available != nullptr) {
        if (cudaFree(d_available) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_available_gpu]: %s",
                    "Failed to allocate device-side memory for the available actions.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    d_available = nullptr;

    return NOVA_SUCCESS;
}

int pomdp_initialize_nonzero_beliefs_gpu(unsigned int r, unsigned int maxNonZeroBeliefs,
        const int *nonZeroBeliefs, int *&d_nonZeroBeliefs)
{
    // Ensure the data is valid.
    if (r == 0 || maxNonZeroBeliefs == 0 || nonZeroBeliefs == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_nonzero_beliefs_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_nonZeroBeliefs, r * maxNonZeroBeliefs * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_nonzero_beliefs_gpu]: %s",
                "Failed to allocate device-side memory for the non-zero belief states.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_nonZeroBeliefs, nonZeroBeliefs, r * maxNonZeroBeliefs * sizeof(int),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_nonzero_beliefs_gpu]: %s",
                "Failed to copy memory from host to device for the non-zero belief states.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}

int pomdp_uninitialize_nonzero_beliefs_gpu(int *&d_nonZeroBeliefs)
{
    if (d_nonZeroBeliefs != nullptr) {
        if (cudaFree(d_nonZeroBeliefs) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_nonzero_beliefs_gpu]: %s",
                    "Failed to allocate device-side memory for the non-zero belief states.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    d_nonZeroBeliefs = nullptr;

    return NOVA_SUCCESS;
}

int pomdp_initialize_successors_gpu(unsigned int n, unsigned int m, unsigned int maxSuccessors,
        const int *successors, int *&d_successors)
{
    // Ensure the data is valid.
    if (n == 0 || m == 0 || maxSuccessors <= 0 || successors == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_successors_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_successors, n * m * maxSuccessors * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_successors_gpu]: %s",
                "Failed to allocate device-side memory for the successor states.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_successors, successors, n * m * maxSuccessors * sizeof(int),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_successors_gpu]: %s",
                "Failed to copy memory from host to device for the successor states.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}

int pomdp_uninitialize_successors_gpu(int *&d_successors)
{
    if (d_successors != nullptr) {
        if (cudaFree(d_successors) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_successors_gpu]: %s",
                    "Failed to allocate device-side memory for the successor states.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    d_successors = nullptr;

    return NOVA_SUCCESS;
}

