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


int pomdp_initialize_nonzero_beliefs_gpu(unsigned int r, unsigned int rz,
        const int *Z, int *&d_Z)
{
    // Ensure the data is valid.
    if (r == 0 || rz == 0 || Z == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_nonzero_beliefs_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_Z, r * rz * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_nonzero_beliefs_gpu]: %s",
                "Failed to allocate device-side memory for the non-zero belief states.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_Z, Z, r * rz * sizeof(int),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_nonzero_beliefs_gpu]: %s",
                "Failed to copy memory from host to device for the non-zero belief states.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int pomdp_uninitialize_nonzero_beliefs_gpu(int *&d_Z)
{
    if (d_Z != nullptr) {
        if (cudaFree(d_Z) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_nonzero_beliefs_gpu]: %s",
                    "Failed to allocate device-side memory for the non-zero belief states.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    d_Z = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_initialize_belief_points_gpu(unsigned int r, unsigned int rz, const float *B,
        float *&d_B)
{
    // Ensure the data is valid.
    if (r == 0 || rz == 0 || B == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_belief_points_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_B, r * rz * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_belief_points_gpu]: %s",
                "Failed to allocate device-side memory for the belief points.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_B, B, r * rz * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
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


int pomdp_initialize_successors_gpu(unsigned int n, unsigned int m, unsigned int ns,
        const int *S, int *&d_S)
{
    // Ensure the data is valid.
    if (n == 0 || m == 0 || ns == 0 || S == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_successors_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_S, n * m * ns * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_successors_gpu]: %s",
                "Failed to allocate device-side memory for the successor states.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_S, S, n * m * ns * sizeof(int),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_successors_gpu]: %s",
                "Failed to copy memory from host to device for the successor states.");
        return NOVA_ERROR_MEMCPY_TO_DEVICE;
    }

    return NOVA_SUCCESS;
}


int pomdp_uninitialize_successors_gpu(int *&d_S)
{
    if (d_S != nullptr) {
        if (cudaFree(d_S) != cudaSuccess) {
            fprintf(stderr, "Error[pomdp_uninitialize_successors_gpu]: %s",
                    "Failed to allocate device-side memory for the successor states.");
            return NOVA_ERROR_DEVICE_FREE;
        }
    }
    d_S = nullptr;

    return NOVA_SUCCESS;
}


int pomdp_initialize_state_transitions_gpu(unsigned int n, unsigned int m, unsigned int ns,
        const float *T, float *&d_T)
{
    // Ensure the data is valid.
    if (n == 0 || m == 0 || ns == 0 || T == nullptr) {
        fprintf(stderr, "Error[pomdp_initialize_state_transitions_gpu]: %s", "Invalid input.");
        return NOVA_ERROR_INVALID_DATA;
    }

    // Allocate the memory on the device.
    if (cudaMalloc(&d_T, n * m * ns * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error[pomdp_initialize_state_transitions_gpu]: %s",
                "Failed to allocate device-side memory for the state transitions.");
        return NOVA_ERROR_DEVICE_MALLOC;
    }

    // Copy the data from the host to the device.
    if (cudaMemcpy(d_T, T, n * m * ns * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
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

