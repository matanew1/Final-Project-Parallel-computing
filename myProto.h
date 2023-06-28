#pragma once

#define BLOCK_SIZE 256
#define CONSTRAINTS 3   

# ifndef POINT_H
# define POINT_H

typedef struct Point {
    int id;
    double x1, x2, a, b;
} Point;


#endif

void test(int *data, int n);
void computeOnGPU(int *N, int *K, double *D, int *tCountSize, double *myTValues, Point *points, int *results);
void writeOutputFile(const char* filename, int tCount, int* results, Point* points, int N);
void gatherResults(int rank, int size, int N, int tCount, int tCountSize, int *results, int *global_results);
void calculateTValues(int tCount, double **tValues);
void readInputFile(const char *filename, int *N, int *K, double *D, int *tCount, Point **points);