#pragma once

#define BLOCK_SIZE 256

# ifndef POINT_H
# define POINT_H

typedef struct Point {
    int id;
    double x1, x2, a, b;
} Point;

#endif

void test(int *data, int n);
int computeOnGPU(int *N, int *K, double *D, int *tCount, int *myPointsCount, double *tValues, int *results, int *resultsCount, int *maxResults, Point *myPoints);
