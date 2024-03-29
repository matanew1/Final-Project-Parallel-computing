#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "myProto.h"

/**
 * Main function
 */
int main(int argc, char *argv[])
{
    double startTime, endTime;
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // start time
    startTime = MPI_Wtime();

    const char *filename = "input.txt"; // Predefined filename

    int N, K, tCount;
    double D;
    Point *points = NULL;

    // Read input file on root process and broadcast values to all processes
    if (rank == 0)
    {
        readInputFile(filename, &N, &K, &D, &tCount, &points);
        if (tCount < size)
        {
            printf("tCount must be greater than or equal to size!\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);      // Broadcast N to all processes
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);      // Broadcast K to all processes
    MPI_Bcast(&D, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);   // Broadcast D to all processes
    MPI_Bcast(&tCount, 1, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast tCount to all processes

    // Define MPI_POINT datatype
    MPI_Datatype MPI_POINT;
    MPI_Type_contiguous(sizeof(Point), MPI_BYTE, &MPI_POINT); // Create custom datatype for Point struct
    MPI_Type_commit(&MPI_POINT);                              // Commit the datatype

    // Allocate memory for points if not the root process
    if (rank != 0)
    {
        points = (Point *)malloc(N * sizeof(Point)); // Allocate memory for points
        if (!points)
        {
            fprintf(stderr, "Failed to allocate points.\n");
            MPI_Finalize();
            exit(1);
        }
    }

    // Broadcast points array from root to all processes
    MPI_Bcast(points, N, MPI_POINT, 0, MPI_COMM_WORLD);

    double *tValues = NULL;
    if (rank == 0)
    {
        // Calculate t values on the root process
        calculateTValues(tCount, &tValues);
    }

    // Calculate sendcounts and displacements for scatterv operation
    int tCountSize = tCount / size;
    int remainingTValues = tCount % size;
    int *sendcounts = (int *)malloc(size * sizeof(int)); // Allocate memory for sendcounts
    int *displs = (int *)malloc(size * sizeof(int));     // Allocate memory for displacements

    // Calculate the sendcounts and displacements
    for (int i = 0; i < size; i++)
    {
        sendcounts[i] = (i < remainingTValues) ? tCountSize + 1 : tCountSize;         // Determine the sendcount for each process
        displs[i] = i * tCountSize + ((i < remainingTValues) ? i : remainingTValues); // Determine the displacement for each process
    }

    int myTValuesSize = sendcounts[rank];
    double *myTValues = (double *)malloc(myTValuesSize * sizeof(double)); // Allocate memory for t values of each process

    // Scatter t values to all processes
    MPI_Scatterv(tValues, sendcounts, displs, MPI_DOUBLE, myTValues, myTValuesSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate memory for results array
    int *results = (int *)malloc(CONSTRAINTS * myTValuesSize * sizeof(int));
    memset(results, -1, CONSTRAINTS * myTValuesSize * sizeof(int)); // Initialize results array with -1 values

    // Compute results on GPU
    computeOnGPU(N, K, D, myTValuesSize, myTValues, points, results);  

    int *global_results = NULL;
    if (rank == 0)
    {
        // Allocate memory for global results on root process
        global_results = (int *)malloc(CONSTRAINTS * tCount * sizeof(int));
        memset(global_results, -1, CONSTRAINTS * tCount * sizeof(int)); // Initialize global results array with -1 values
    }

    // Gather results from all processes to root process
    gatherResults(rank, size, N, tCount, myTValuesSize, results, global_results);

    // Write output file on root process
    if (rank == 0)
    {
        writeOutputFile("output.txt", tValues, tCount, global_results, points, N);
        
        // end time
        endTime = MPI_Wtime();
        double executionTime = endTime - startTime;

        printf("Parallel - Execution time: %f seconds\n", executionTime);
    }

    // Deallocate memory
    free(global_results);
    free(points);
    free(tValues);
    MPI_Type_free(&MPI_POINT);
    free(sendcounts);
    free(displs);
    free(myTValues);
    free(results);

    MPI_Finalize();
    return 0;
}
