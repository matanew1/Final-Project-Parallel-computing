#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

/**
 * Calculate the distance between two points.
 * @param p1 Struct that points to one point.
 * @param p2 Struct that points to the second point.
 * @param t The current t-value.
 * @return The distance between the two points.
 */
__device__ double calcDistance(const Point p1, const Point p2, double t)
{
    double x1 = ((p1.x2 - p1.x1) / 2) * __sinf(t * M_PI / 2) + ((p1.x2 + p1.x1) / 2);
    double y1 = p1.a * x1 + p1.b;

    double x2 = ((p2.x2 - p2.x1) / 2) * __sinf(t * M_PI / 2) + (p2.x2 + p2.x1) / 2;
    double y2 = p2.a * x2 + p2.b;

    double dx = x2 - x1;
    double dy = y2 - y1;

    return sqrt(dx * dx + dy * dy);
}

/**
 * Update proximity points in an atomic manner.
 * @param tIdx The current t-value in the round.
 * @param results Array of results.
 * @param pointId A point that satisfies the condition.
 */
__device__ void updateResults(int tIdx, int *results, int pointId)
{
    // Loop through each constraint
    for (int i = 0; i < CONSTRAINTS; i++)
    {
        // Calculate the index in the results array for the current tIdx and constraint
        int index = tIdx * CONSTRAINTS + i;

        // Retrieve the current value stored at the calculated index
        int currentVal = results[index];

        // Check if the current value is -1 (indicating an unset value)
        if (currentVal == -1)
        {
            // Attempt to atomically update the value at the calculated index
            // If the current value is still -1, the swap is performed and returns true
            if (atomicCAS(&results[index], currentVal, pointId) == currentVal)
            {
                // If the swap was successful (i.e., the current value was still -1),
                // exit the loop and the function
                return;
            }
        }
    }
}

/**
 * Check proximity of points on the GPU.
 * @param d_points Array of all N points.
 * @param N Number of points.
 * @param tValue Current t-value.
 * @param D Max distance to check.
 * @param d_results Array of results.
 * @param K Need at least K points that fulfill the condition of Proximity Criteria.
 * @param tIdx The current t in the for loop.
 */
__global__ void checkProximity(Point *d_points, int N, double tValue, double D, int *d_results, double K, int tIdx)
{
    // Calculate the unique thread ID within the grid
    int pid = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize a counter to track the number of nearby points
    int counter = 0;

    // Check if the thread ID is within the valid range of points
    if (pid < N)
    {
        // Loop through all points to check proximity
        for (int i = 0; i < N; i++)
        {
            // Check if the proximity results have been flagged as complete
            if (atomicAdd(&d_results[tIdx * CONSTRAINTS + CONSTRAINTS - 1], 0) != -1)
                return; // If the results are complete, exit the function immediately

            // Compare the ID of the current point and the checked point
            Point current = d_points[pid];
            Point checked = d_points[i];

            if (checked.id != current.id && calcDistance(current, checked, tValue) < D)
            {
                // Increment the counter when the condition is met
                counter++;

                // Check if the required number of nearby points have been found
                if (counter == K)
                {
                    // Update the results array with the current point's ID
                    updateResults(tIdx, d_results, current.id);
                    // Exit the loop since the required condition has been satisfied
                    break;
                }
            }
        }
    }
}

/**
 * Allocate memory on the GPU.
 * @param ptr Pointer to the memory to be allocated.
 * @param size Size of the memory to allocate (in bytes).
 */
void allocateMemDevice(void **ptr, size_t size)
{
    cudaError_t err = cudaMalloc(ptr, size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cannot to allocate memory on device. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * Copy memory between host and device.
 * @param dest Pointer to the destination in device/host memory.
 * @param src Pointer to the source in host/device memory.
 * @param size How much data needs to be copied (in bytes).
 * @param direction Which direction to copy the memory (host->device) or (device->host).
 */
void copyMemory(void *dest, void *src, size_t size, cudaMemcpyKind direction)
{
    /*Copy mem from device to host OR host to device depending on the direction*/
    cudaError_t err = cudaMemcpy(dest, src, size, direction);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


/**
 * Perform computation on the GPU.
 * @param N Number of points.
 * @param K Need at least K points that fulfill the condition of Proximity Criteria.
 * @param D Max distance to check.
 * @param tCount Number of t values.
 * @param tValues Array of t values.
 * @param points Array of points.
 * @param results Array of results.
 * @return 0 if the computation is successful.
 */
int computeOnGPU(int N, int K, double D, int tCount, double *tValues, Point *points, int *results)
{
    cudaError_t err = cudaSuccess;

    int threadPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadPerBlock - 1) / threadPerBlock;
    Point *d_points = NULL;
    int *d_results = NULL;

    /*Allocating mem on gpu section*/
    allocateMemDevice((void **)&d_results, CONSTRAINTS * tCount * sizeof(int)); 
    allocateMemDevice((void **)&d_points, N * sizeof(Point));
    /*End allocate mem*/

    /*Copy mem to Device section*/
    copyMemory(d_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
    copyMemory(d_results, results, tCount * CONSTRAINTS * sizeof(int), cudaMemcpyHostToDevice);
    /*End copy mem to Device*/

    /*for each tvalue we will send it to GPU to compute the data and save it on results array*/
    for (int i = 0; i < tCount; i++)
    {
        checkProximity<<<blocksPerGrid, threadPerBlock>>>(d_points, N, tValues[i], D, d_results, K, i);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to lanch checkProximity kernel. -%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    err = cudaMemcpy(results, d_results, tCount * CONSTRAINTS * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data. -%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if (cudaFree(d_points) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if (cudaFree(d_results) != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device data - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return 0;
}
