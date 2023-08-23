#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__device__ double calculateSin(double t) {
    double angle = t * M_PI / 2;
    return __sinf(angle); // Using hardware-accelerated sin
}

__device__ double calculateX(double x1, double x2, double t) {
    double sinValue = calculateSin(t); // Calculate sin only once
    return ((x2 - x1) / 2) * sinValue + ((x2 + x1) / 2);
}

__device__ bool isProximityCriteriaMet(const Point *p1, const Point *p2, double t, double D) {
    double x1 = calculateX(p1->x1, p1->x2, t);
    double y1 = p1->a * x1 + p1->b;
    
    double x2 = calculateX(p2->x1, p2->x2, t);
    double y2 = p2->a * x2 + p2->b;

    double dx = x2 - x1; // Calculate dx and dy only once
    double dy = y2 - y1;

    double distanceSquared = dx * dx + dy * dy; // Avoiding square root for Euclidean distance check

    return distanceSquared <= D * D; // Compare with squared threshold
}

__global__ void checkProximityCriteria(Point *points, double *tValues, const int tCount, const int N, const int K, const double D, int *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tCount)
        return;

    double t = tValues[idx];

    int pointId = -1;

    for (int i = 0; i < N; i++) {
        int count = 0; // Initialize count inside the loop

        for (int j = 0; j < N; j++) {
            if (i != j && isProximityCriteriaMet(&points[i], &points[j], t, D)) {
                if (++count == K) { // Increment and check in one step
                    pointId = points[i].id;
                    break;
                }
            }
        }

        if (pointId != -1)
            break;
    }

    for (int j = 0; j < CONSTRAINTS; j++) {
        int targetIndex = idx * CONSTRAINTS + j;

        if (results[targetIndex] == -1) {
            atomicExch(&results[targetIndex], pointId);
            return;
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

    checkProximityCriteria<<<blocksPerGrid, threadPerBlock>>>(d_points, d_tValues, tCountSize, N, K, D, d_results);
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
