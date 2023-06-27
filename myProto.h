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
