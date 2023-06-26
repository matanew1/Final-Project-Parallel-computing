#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

__device__ double calcDistance(const Point* p1, const Point* p2, double* t) {
    double x1 = ((p1->x2 - p1->x1) / 2) * sin((*t) * M_PI / 2) + ((p1->x2 + p1->x1) / 2);
    double y1 = p1->a * x1 + p1->b;

    double x2 = ((p2->x2 - p2->x1) / 2) * sin((*t) * M_PI / 2) + ((p2->x2 + p2->x1) / 2);
    double y2 = p2->a * x2 + p2->b;

    double distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));

    return distance;
}

__global__ void checkProximityCriteria(int* count, const Point *points, double *tValues, const int tCount,const int N,const int K, const double D, double* distances)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx

    if (idx < tCount)
    {       
        double* currentTValue = &(tValues[idx]);
        for (int i = 0; i < N; i++) {
            const Point currentPoint = points[i];
            for(int j = 0; j < N && j != i; j++) {
                const Point otherPoint = points[j];
                double distance = calcDistance(&currentPoint, &otherPoint, currentTValue);

                if (distance <= D) {
                    atomicAdd(count, 1);
                    if ((*count) >= K) {
                        break;
                    }
                }
                distances[idx * N * (N - 1) + i * (N - 1) + j] = distance;
            }
        }
    }
}


void computeOnGPU(int *N, int *K, double *D, int *tCountSize, double *myTValues, Point *points) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int blocksPerGrid = ((*tCountSize) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int threadsPerBlock = BLOCK_SIZE;

    Point *dPoints;         // points for device
    double *dTValues;       // tValues for device
    double *dDistances;     // distances array for device

    err = cudaMalloc(&dPoints, (*N) * sizeof(Point));
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&dTValues, (*tCountSize) * sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&dDistances, (*tCountSize) * (*N) * (*N - 1) * sizeof(double));
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(dPoints, points, (*N) * sizeof(Point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(dTValues, myTValues, (*tCountSize) * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int count = 0;
    checkProximityCriteria<<<blocksPerGrid, threadsPerBlock>>>(&count, dPoints, dTValues, *tCountSize, *N, *K, *D, dDistances);

    double *distances = (double*)malloc((*tCountSize) * (*N) * (*N - 1) * sizeof(double));
    if (distances == NULL){
        fprintf(stderr, "Error allocating host memory!\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(distances, dDistances, (*tCountSize) * (*N) * (*N - 1) * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        fprintf(stderr, "Error in line %d (error code %s)!\n", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print the distances
    for (int idx = 0; idx < (*tCountSize); idx++) {
        for (int i = 0; i < (*N); i++) {
            for (int j = 0; j < (*N - 1); j++) {
                printf("%d) Point %d and point %d - distance %lf\n", idx, i, j, distances[idx * (*N) * (*N - 1) + i * (*N - 1) + j]);
            }
        }
    }

    cudaFree(dPoints);
    cudaFree(dTValues);
    cudaFree(dDistances);
    free(distances);
}
