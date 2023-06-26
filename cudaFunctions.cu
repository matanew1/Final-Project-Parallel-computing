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

__global__ void checkProximityCriteria(int* count, const Point *points, double *tValues, const int tCount,const int N,const int K, const double D){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // point idx

    if (idx < tCount)
    {
        
        double* t = &(tValues[idx]);
        for (int i = 0; i < N; i++) {
            for(int j = 0; j < N && j != i; j++) {
                double distance = calcDistance(&points[i], &points[j], t);

                if (distance <= D) {
                    (*count)++;
                    if ((*count) >= K) {
                        break;
                    }
                }  
            }         
        }
    }
}


void computeOnGPU(int *N, int *K, double *D, int *tCountSize, double *myTValues, Point *points) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int blocksPerGrid = ((*tCountSize) + (*N)) / BLOCK_SIZE < 1 ? 1 : ((*tCountSize) + (*N)) / BLOCK_SIZE < 1; 
    int threadsPerBlock = BLOCK_SIZE;

    Point *dPoints;     // point for device
    double *dTValues;   // tValues for device    

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

    checkProximityCriteria<<<blocksPerGrid, threadsPerBlock>>>(&count, dPoints, dTValues, *tCountSize, *N, *K, *D);
    printf("Count: %d\n",count);
    
    // free device allocation
    cudaFree(dPoints);
    cudaFree(dTValues);
}
