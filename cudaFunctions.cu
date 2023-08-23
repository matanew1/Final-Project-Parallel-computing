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
    double x1 = ((p1->x2 - p1->x1) / 2) * __sinf((*t) * M_PI / 2) + ((p1->x2 + p1->x1) / 2);

    // Calculate the y-coordinate of the first point based on the equation of a line.
    // y1 = a * x1 + b;
    double y1 = p1->a * x1 + p1->b;

    // Calculate the x-coordinate of the second point at a given value of t.
    // x2 = ((x2 - x1) / 2) * sin(t * π / 2) + ((x2 + x1) / 2);
    double x2 = ((p2->x2 - p2->x1) / 2) * __sinf((*t) * M_PI / 2) + ((p2->x2 + p2->x1) / 2);

    // Calculate the y-coordinate of the second point based on the equation of a line.
    // y2 = a * x2 + b;
    double y2 = p2->a * x2 + p2->b;

    // Calculate the distance between the two points using the Euclidean distance formula.
    double dx = x2 - x1; // Calculate dx and dy only once
    double dy = y2 - y1;

    double distanceSquared = dx * dx + dy * dy; // Avoiding square root for

    // Return the calculated distance.
    return sqrt(distanceSquared);
}

/**
 * Check if the proximity criteria is met between two points at a given t value.
 *
 * @param p1  Pointer to the first point
 * @param p2  Pointer to the second point
 * @param t   Pointer to the t value
 * @param D   Proximity criteria threshold
 * @return    True if the criteria is met, false otherwise
 */
__device__ bool isProximityCriteriaMet(const Point *p1, const Point *p2, double *t, double D)
{
    // Calculate the distance between the two points using the calcDistance function.
    double distance = calcDistance(p1, p2, t);

    // Check if the calculated distance is less than or equal to the proximity criteria D.
    // If the distance is within the criteria, return true. Otherwise, return false.
    return distance <= D;
}

/**
 * Update the results array with the proximity point ID.
 *
 * @param idx              Index of the results array
 * @param results          Pointer to the results array
 * @param proximityPointId Proximity point ID
 */
__device__ void updateResults(int idx, int *results, int proximityPointId)
{
    // Iterate over the CONSTRAINTS number of times.
    for (int j = 0; j < CONSTRAINTS; j++)
    {
        // Calculate the target index based on the current index and j.
        int targetIndex = idx * CONSTRAINTS + j;

        // Check if the value at the target index is -1.
        // If it is, update the value using atomicExch to avoid race conditions.
        // atomicExch performs an atomic exchange operation and returns the original value.
        // If the original value is -1, replace it with proximityPointId and return.
        if (results[targetIndex] == -1)
        {
            atomicExch(&results[targetIndex], proximityPointId);
            return;
        }
    }
}

/**
 * GPU kernel function to check the proximity criteria for points and update the results array.
 *
 * @param points     Array of points
 * @param tValues    Array of t values
 * @param tCount     Total number of t values
 * @param N          Number of points
 * @param K          Number of points to satisfy proximity criteria
 * @param D          Proximity criteria threshold
 * @param results    Results array
 */
__global__ void checkProximityCriteria(Point *points, double *tValues, const int tCount, const int N, const int K, const double D, int *results)
{
    // Calculate the index of the current thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx

    // Check if the index is greater than or equal to tCount.
    // If it is, return as it is outside the valid range.
    if (idx >= tCount)
        return; // specific t

    // Get the current value of t from tValues.
    double t = tValues[idx];

    // Variable to keep track of the count of proximity points.
    int count = 0;

    // Iterate over the points array.
    for (int i = 0; i < N; i++)
    {
        count = 0;

        // Check proximity for each point with all other points.
        for (int j = 0; j < N; j++)
        {
            // Check if the points are different and meet the proximity criteria.
            if (i != j && isProximityCriteriaMet(&points[i], &points[j], &t, D))
            {
                count++;

                // Check if the count of proximity points reaches the desired value K.
                if (count == K)
                {
                    // printf("Found proximity criteria point\n");
                    // Get the ID of the proximity point.
                    int proximityPointId = points[i].id;

                    // printf("Start update resutls\n");
                    // Update the results array with the proximity point ID.
                    updateResults(idx, results, proximityPointId);
                    // printf("End updated resutls\n");

                    // Break out of the inner loop since K proximity points have been found.
                    break;
                }
            }
        }
    }
}

/**
 * Allocate device memory.
 *
 * @param devPtr  Pointer to the allocated device memory
 * @param size    Size of the memory to allocate
 */
void allocateDeviceMemory(void **devPtr, size_t size)
{
    // Allocate device memory using cudaMalloc.
    cudaError_t err = cudaMalloc(devPtr, size);

    // Check if there was an error during memory allocation.
    // If an error occurred, print the error message and exit the program.
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Copy data from host to device.
 *
 * @param dst    Pointer to the destination in device memory
 * @param src    Pointer to the source in host memory
 * @param count  Number of bytes to copy
 * @param kind   Type of cudaMemcpy operation
 */
void copyHostToDevice(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    // Copy data from the host to the device using cudaMemcpy.
    cudaError_t err = cudaMemcpy(dst, src, count, kind);

    // Check if there was an error during data copy.
    // If an error occurred, print the error message and exit the program.
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Copy data from device to host.
 *
 * @param dst    Pointer to the destination in host memory
 * @param src    Pointer to the source in device memory
 * @param count  Number of bytes to copy
 * @param kind   Type of cudaMemcpy operation
 */
void copyDeviceToHost(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    // Copy data from the device to the host using cudaMemcpy.
    cudaError_t err = cudaMemcpy(dst, src, count, kind);

    // Check if there was an error during data copy.
    // If an error occurred, print the error message and exit the program.
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Free device memory.
 *
 * @param devPtr  Pointer to the device memory to free
 */
void freeDeviceMemory(void *devPtr)
{
    // Free the device memory using cudaFree.
    cudaFree(devPtr);
}
/**
 * Compute the proximity criteria on the GPU.
 *
 * @param N            Number of points
 * @param K            Number of points to satisfy proximity criteria
 * @param D            Proximity criteria threshold
 * @param tCountSize   Size of the t values for each process
 * @param myTValues    T values for each process
 * @param points       Array of points
 * @param results      Results array
 */
void computeOnGPU(int *N, int *K, double *D, int *tCountSize, double *myTValues, Point *points, int *results)
{
    cudaError_t err = cudaSuccess;

    // Determine the number of threads per block based on the BLOCK_SIZE constant
    // and the size of tCount.
    int threadPerBlock = min(BLOCK_SIZE, *tCountSize);

    // Calculate the number of blocks needed to cover all tCount values.
    int blocksPerGrid = (*tCountSize + threadPerBlock - 1) / threadPerBlock;

    // Declare device pointers for points, tValues, and results.
    Point *d_points = NULL;
    double *d_tValues = NULL;
    int *d_results = NULL;

    // Allocate device memory for points, tValues, and results.
    allocateDeviceMemory((void **)&d_points, (*N) * sizeof(Point));
    allocateDeviceMemory((void **)&d_tValues, (*tCountSize) * sizeof(double));
    allocateDeviceMemory((void **)&d_results, CONSTRAINTS * (*tCountSize) * sizeof(int));
    // printf("Allocated device memory for points, tValues, and results\n");

    // Copy points, tValues, and results from the host to the device.
    copyHostToDevice(d_points, points, (*N) * sizeof(Point), cudaMemcpyHostToDevice);
    copyHostToDevice(d_tValues, myTValues, (*tCountSize) * sizeof(double), cudaMemcpyHostToDevice);
    copyHostToDevice(d_results, results, CONSTRAINTS * (*tCountSize) * sizeof(int), cudaMemcpyHostToDevice);
    // printf("Copy host memory to device for points, tValues, and results\n");

    // Launch the checkProximityCriteria kernel on the device.
    checkProximityCriteria<<<blocksPerGrid, threadPerBlock>>>(d_points, d_tValues, *tCountSize, *N, *K, *D, d_results);

    // Check if there was an error launching the kernel.
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch checkProximityCriteria kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Synchronize the device to ensure all kernel calls are completed.
    cudaDeviceSynchronize();

    // Copy results from the device to the host.
    copyDeviceToHost(results, d_results, CONSTRAINTS * (*tCountSize) * sizeof(int), cudaMemcpyDeviceToHost);


    // Free device memory for points, tValues, and results.
    freeDeviceMemory(d_points);
    freeDeviceMemory(d_tValues);
    freeDeviceMemory(d_results);
}