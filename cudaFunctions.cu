#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

/**
 * Calculate the distance between two points at a given t value.
 *
 * @param p1  Pointer to the first point
 * @param p2  Pointer to the second point
 * @param t   Pointer to the t value
 * @return    The calculated distance
 */
__device__ double calcDistance(const Point *p1, const Point *p2, double *t)
{
    // Calculate the x-coordinate of the first point at a given value of t.
    // x1 = ((x2 - x1) / 2) * sin(t * π / 2) + ((x2 + x1) / 2);
    double x1 = ((p1->x2 - p1->x1) / 2) * __fsin((*t) * M_PI / 2) + ((p1->x2 + p1->x1) / 2);

    // Calculate the y-coordinate of the first point based on the equation of a line.
    // y1 = a * x1 + b;
    double y1 = p1->a * x1 + p1->b;

    // Calculate the x-coordinate of the second point at a given value of t.
    // x2 = ((x2 - x1) / 2) * sin(t * π / 2) + ((x2 + x1) / 2);
    double x2 = ((p2->x2 - p2->x1) / 2) * __fsin((*t) * M_PI / 2) + ((p2->x2 + p2->x1) / 2);

    // Calculate the y-coordinate of the second point based on the equation of a line.
    // y2 = a * x2 + b;
    double y2 = p2->a * x2 + p2->b;

    // Calculate the distance between the two points using the Euclidean distance formula.
    // distance = sqrt((x2 - x1)^2 + (y2 - y1)^2);
    double distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));

    double distanceSquared = dx * dx + dy * dy; // Avoiding square root for Euclidean distance check

    return distanceSquared <= D * D; // Compare with squared threshold
}

__global__ void checkProximityCriteria(Point *points, double *tValues, const int tCount, const int N, const int K, const double D, int *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tCount)
        return;

    double t = tValues[idx];
    __shared__ int sharedResults[BLOCK_SIZE * CONSTRAINTS];

    for (int j = threadIdx.x; j < CONSTRAINTS; j += blockDim.x) {
        sharedResults[threadIdx.x * CONSTRAINTS + j] = -1;
    }

    __syncthreads();

    int pointId = -1;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int count = 0;

        for (int j = 0; j < N; j++) {
            if (i != j && isProximityCriteriaMet(&points[i], &points[j], &t, D)) {
                count++;

                if (count == K) {
                    pointId = points[i].id;
                    break;
                }
            }
        }

        if (pointId != -1) {
            break;
        }
    }

    // Update sharedResults
    for (int j = threadIdx.x; j < CONSTRAINTS; j += blockDim.x) {
        if (pointId != -1 && sharedResults[j] == -1) {
            sharedResults[j] = pointId;
        }
    }

    __syncthreads();

    // Have one thread per block write sharedResults to global results
    if (threadIdx.x == 0) {
        for (int j = 0; j < CONSTRAINTS; j++) {
            int targetIndex = idx * CONSTRAINTS + j;
            results[targetIndex] = sharedResults[j];
        }
    }
}

void computeOnGPU(int N, int K, double D, int tCountSize, double *myTValues, Point *points, int *results) {
    cudaError_t err = cudaSuccess;
    int threadPerBlock = min(BLOCK_SIZE, tCountSize);
    int blocksPerGrid = (tCountSize + threadPerBlock - 1) / threadPerBlock;

    Point *d_points = nullptr;
    double *d_tValues = nullptr;
    int *d_results = nullptr;

    allocateDeviceMemory((void **)&d_points, N * sizeof(Point));
    allocateDeviceMemory((void **)&d_tValues, tCountSize * sizeof(double));
    allocateDeviceMemory((void **)&d_results, CONSTRAINTS * tCountSize * sizeof(int));

    copyHostToDevice(d_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
    copyHostToDevice(d_tValues, myTValues, tCountSize * sizeof(double), cudaMemcpyHostToDevice);
    copyHostToDevice(d_results, results, CONSTRAINTS * tCountSize * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate device memory for points, tValues, and results.
    allocateDeviceMemory((void **)&d_points, (*N) * sizeof(Point));
    allocateDeviceMemory((void **)&d_tValues, (*tCountSize) * sizeof(double));
    allocateDeviceMemory((void **)&d_results, CONSTRAINTS * (*tCountSize) * sizeof(int));
    printf("Allocated device memory for points, tValues, and results\n");

    // Copy points, tValues, and results from the host to the device.
    copyHostToDevice(d_points, points, (*N) * sizeof(Point), cudaMemcpyHostToDevice);
    copyHostToDevice(d_tValues, myTValues, (*tCountSize) * sizeof(double), cudaMemcpyHostToDevice);
    copyHostToDevice(d_results, results, CONSTRAINTS * (*tCountSize) * sizeof(int), cudaMemcpyHostToDevice);
    printf("Copy host memory to device for points, tValues, and results\n");

    // Launch the checkProximityCriteria kernel on the device.
    checkProximityCriteria<<<blocksPerGrid, threadPerBlock>>>(d_points, d_tValues, *tCountSize, *N, *K, *D, d_results);

    // Check if there was an error launching the kernel.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    copyDeviceToHost(results, d_results, CONSTRAINTS * tCountSize * sizeof(int), cudaMemcpyDeviceToHost);

    freeDeviceMemory(d_points);
    freeDeviceMemory(d_tValues);
    freeDeviceMemory(d_results);
}
