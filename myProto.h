#pragma once

// Define constant values
#define BLOCK_SIZE 256   // The block size used for GPU computations
#define CONSTRAINTS 3    // The number of constraints to satisfy
#define true 1           // Boolean true value
#define false 0          // Boolean false value

# ifndef POINT_H
# define POINT_H

// Define the Point structure
typedef struct Point {
    int id;             // Point ID
    double x1, x2, a, b; // Point attributes
} Point;

#endif

// Function prototypes
void computeOnGPU(int *N, int *K, double *D, int *tCountSize, double *myTValues, Point *points, int *results);
/**
 * Perform computations on GPU for proximity criteria checking.
 *
 * @param N           Pointer to the number of points
 * @param K           Pointer to the number of points to satisfy proximity criteria
 * @param D           Pointer to the proximity criteria threshold
 * @param tCountSize  Pointer to the size of t values for the current process
 * @param myTValues   Pointer to the t values of the current process
 * @param points      Pointer to the array of points
 * @param results     Pointer to the results array
 */

void writeOutputFile(const char* filename, double* tValues, int tCount, int* results, Point* points, int N);
/**
 * Write the output file with points that satisfy the proximity criteria.
 *
 * @param filename  Output file name
 * @param tCount    Total number of t values
 * @param results   Results array
 * @param points    Array of points
 * @param N         Number of points
 */

void gatherResults(int rank, int size, int N, int tCount, int tCountSize, int *results, int *global_results);
/**
 * Gather results from all processes to the root process.
 *
 * @param rank           Rank of the current process
 * @param size           Total number of processes
 * @param N              Number of points
 * @param tCount         Total number of t values
 * @param tCountSize     Size of t values for the current process
 * @param results        Results array of the current process
 * @param global_results Global results array in the root process
 */

void calculateTValues(int tCount, double **tValues);
/**
 * Calculate t values based on the given tCount.
 *
 * @param tCount    Total number of t values
 * @param tValues   Pointer to the t values array
 */

void readInputFile(const char *filename, int *N, int *K, double *D, int *tCount, Point **points);
/**
 * Read input file and populate the variables.
 *
 * @param filename  Input file name
 * @param N         Pointer to the number of points
 * @param K         Pointer to the number of points to satisfy proximity criteria
 * @param D         Pointer to the proximity criteria threshold
 * @param tCount    Pointer to the total number of t values
 * @param points    Pointer to the array of points
 */

