#include "myProto.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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
void readInputFile(const char *filename, int *N, int *K, double *D, int *tCount, Point **points)
{
   // Open the input file in read mode
   FILE *file = fopen(filename, "r");
   if (!file)
   {
      fprintf(stderr, "Failed to open input file.\n");
      MPI_Finalize();
      exit(1);
   }

   // Read the required values from the input file
   if (fscanf(file, "%d %d %lf %d\n", N, K, D, tCount) != 4)
   {
      fprintf(stderr, "Failed to read required values from input file.\n");
      fclose(file);
      MPI_Finalize();
      exit(1);
   }

   // Allocate memory for the points array
   *points = (Point *)malloc(*N * sizeof(Point));
   if (!(*points))
   {
      fprintf(stderr, "Failed to allocate points.\n");
      fclose(file);
      MPI_Finalize();
      exit(1);
   }

   // Read point data from the input file and populate the points array
   for (int i = 0; i < *N; ++i)
   {
      if (fscanf(file, "%d %lf %lf %lf %lf\n", &((*points)[i].id), &((*points)[i].x1), &((*points)[i].x2), &((*points)[i].a), &((*points)[i].b)) != 5)
      {
         fprintf(stderr, "Failed to read point data from input file.\n");
         fclose(file);
         MPI_Finalize();
         exit(1);
      }
   }

   fclose(file); // Close the input file
}


/**
 * Calculate t values based on the given tCount.
 *
 * @param tCount    Total number of t values
 * @param tValues   Pointer to the t values array
 */
void calculateTValues(int tCount, double **tValues)
{
   *tValues = (double *)malloc((tCount + 1) * sizeof(double)); // Allocate memory for t values array

   // calculate all t points
#pragma omp parallel for
   for (int i = 0; i <= tCount; ++i)
   {
      (*tValues)[i] = (2.0 * i / tCount) - 1; // Calculate the t value for each index
   }
}

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
void gatherResults(int rank, int size, int N, int tCount, int tCountSize, int *results, int *global_results)
{
   int *recvcounts = (int *)malloc(size * sizeof(int));
   int *displs = (int *)malloc(size * sizeof(int));

   // Calculate the recvcounts for the 2D array
   for (int i = 0; i < size; i++)
   {
      recvcounts[i] = CONSTRAINTS * ((i < tCount % size) ? (tCount / size + 1) : (tCount / size));
   }

   // Calculate the displacements for the 2D array
   displs[0] = 0;
   for (int i = 1; i < size; i++)
   {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
   }

   // Gather the 2D array results from all processes into global_results on rank 0
   MPI_Gatherv(results, CONSTRAINTS * tCountSize, MPI_INT,
               global_results, recvcounts, displs, MPI_INT,
               0, MPI_COMM_WORLD);

   free(recvcounts); // Free memory allocated for recvcounts array
   free(displs);    // Free memory allocated for displs array
}

/**
 * Write the output file with points that satisfy the proximity criteria.
 *
 * @param filename  Output file name
 * @param tCount    Total number of t values
 * @param results   Results array
 * @param points    Array of points
 * @param N         Number of points
 */
void writeOutputFile(const char *filename, int tCount, int *results, Point *points, int N)
{
   FILE *file = fopen(filename, "w"); // Open the output file in write mode
   if (!file)
   {
      fprintf(stderr, "Failed to open output file.\n");
      MPI_Finalize();
      exit(1);
   }

   int proximityFound = 0;

   // Use OpenMP parallelism for the outer loop
#pragma omp parallel for
   for (int i = 0; i < tCount; i++)
   {
      int count = 0;
      int pointIDs[3] = {-1, -1, -1};

      for (int j = 0; j < CONSTRAINTS; j++)
      {
         int pointID = results[i * CONSTRAINTS + j];
         if (pointID >= 0 && pointID < N)
         {
            pointIDs[count] = pointID;
            count++;
         }
      }

      if (count == 3)
      {
         proximityFound = 1;

         // Use OpenMP critical section for file writing
#pragma omp critical
         {
            fprintf(file, "Points ");
            for (int j = 0; j < 3; j++)
            {
               fprintf(file, "pointID%d", pointIDs[j]);
               if (j < 2)
                  fprintf(file, ", ");
            }
            fprintf(file, " satisfy Proximity Criteria at t = %.2f\n", 2.0 * i / tCount - 1);
         }
      }
   }

   if (!proximityFound)
   {
      fprintf(file, "There were no 3 points found for any t.\n");
   }

   fclose(file); // Close the output file
}

