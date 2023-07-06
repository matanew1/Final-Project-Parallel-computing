#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "myProto.h"

/**
 * Read input file and populate variables.
 *
 * @param filename  Input file name
 * @param N         Pointer to store the value of N
 * @param K         Pointer to store the value of K
 * @param D         Pointer to store the value of D
 * @param tCount    Pointer to store the value of tCount
 * @param points    Pointer to store the array of points
 */
void readInputFile(const char *filename, int *N, int *K, double *D, int *tCount, Point **points)
{
   FILE *file = fopen(filename, "r"); // Open the input file in read mode
   if (!file)
   {
      fprintf(stderr, "Failed to open input file.\n");
      exit(1);
   }

   if (fscanf(file, "%d %d %lf %d\n", N, K, D, tCount) != 4) // Read N, K, D, and tCount from the file
   {
      fprintf(stderr, "Failed to read required values from input file.\n");
      fclose(file);
      exit(1);
   }

   *points = (Point *)malloc(*N * sizeof(Point)); // Allocate memory for the points array
   if (!(*points))
   {
      fprintf(stderr, "Failed to allocate points.\n");
      fclose(file);
      exit(1);
   }

   for (int i = 0; i < *N; ++i)
   {
      if (fscanf(file, "%d %lf %lf %lf %lf\n", &((*points)[i].id), &((*points)[i].x1), &((*points)[i].x2), &((*points)[i].a), &((*points)[i].b)) != 5) // Read point data from the file
      {
         fprintf(stderr, "Failed to read point data from input file.\n");
         fclose(file);
         exit(1);
      }
   }

   fclose(file); // Close the file
}

/**
 * Calculate t values based on tCount.
 *
 * @param tCount   Total number of t values
 * @param tValues  Pointer to store the array of t values
 */
void calculateTValues(int tCount, double **tValues)
{
   *tValues = (double *)malloc((tCount + 1) * sizeof(double)); // Allocate memory for the t values array
   if (!(*tValues))
   {
      fprintf(stderr, "Failed to allocate t values.\n");
      exit(1);
   }

   for (int i = 0; i <= tCount; ++i)
   {
      (*tValues)[i] = (2.0 * i / tCount) - 1; // Calculate each t value
   }
}

/**
 * Calculate distance between two points at given t.
 *
 * @param p1  Pointer to the first point
 * @param p2  Pointer to the second point
 * @param t   T value
 * @return    Calculated distance
 */
double calcDistance(const Point *p1, const Point *p2, double t)
{
   double x1 = ((p1->x2 - p1->x1) / 2) * sin(t * M_PI / 2) + ((p1->x2 + p1->x1) / 2); // Calculate x1 valuefor the given t
   double y1 = p1->a * x1 + p1->b; // Calculate y1 value

   double x2 = ((p2->x2 - p2->x1) / 2) * sin(t * M_PI / 2) + ((p2->x2 + p2->x1) / 2); // Calculate x2 value
   double y2 = p2->a * x2 + p2->b; // Calculate y2 value

   double distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2)); // Calculate the distance between the two points

   return distance; // Return the calculated distance
}

/**
 * Check if proximity criteria is met between two points at given t and D.
 *
 * @param p1  Pointer to the first point
 * @param p2  Pointer to the second point
 * @param t   T value
 * @param D   Proximity criteria threshold
 * @return    1 if proximity criteria is met, 0 otherwise
 */
int isProximityCriteriaMet(const Point *p1, const Point *p2, double t, double D)
{
   double distance = calcDistance(p1, p2, t); // Calculate the distance between the two points
   return distance <= D; // Check if the distance is less than or equal to the proximity criteria threshold
}

/**
 * Update results array with the proximityPointId at the given index.
 *
 * @param idx                Index in the results array
 * @param results            Pointer to the results array
 * @param proximityPointId   Proximity point ID to update in the results array
 */
void updateResults(int idx, int *results, int proximityPointId)
{
   for (int j = 0; j < CONSTRAINTS; j++)
   {
      int targetIndex = idx * CONSTRAINTS + j; // Calculate the target index in the results array
      if (results[targetIndex] == -1) // Check if the target index is empty
      {
         results[targetIndex] = proximityPointId; // Update the results array with the proximity point ID
         return;
      }
   }
}

/**
 * Check proximity criteria between points for all t values.
 *
 * @param points   Array of points
 * @param N        Number of points
 * @param tValues  Array of t values
 * @param tCount   Number of t values
 * @param D        Proximity criteria threshold
 * @param results  Pointer to the results array
 * @param K        Number of points to satisfy proximity criteria
 */
void checkProximityCriteria(Point *points, int N, double *tValues, int tCount, double D, int *results, int K)
{
   int count = 0;
   for (int i = 0; i < tCount; i++)
   {
      for (int j = 0; j < N; j++)
      {
         count = 0;
         for (int k = 0; k < N; k++)
         {
            if (j != k && isProximityCriteriaMet(&points[j], &points[k], tValues[i], D)) // Check if proximity criteria is met between two points
            {
               count++;
               if (count == K) // Check if the desired number of points to satisfy the proximity criteria is reached
               {
                  int proximityPointId = points[j].id; // Get the ID of the proximity point
                  updateResults(i, results, proximityPointId); // Update the results array
                  break;
               }
            }
         }
      }
  }
}

/**
 * Write output file with points that satisfy the proximity criteria.
 *
 * @param filename      Output file name
 * @param tCount        Total number of t values
 * @param results       Results array
 * @param points        Array of points
 * @param N             Number of points
 */
void writeOutputFile(const char *filename, int tCount, int *results, Point *points, int N)
{
   FILE *file = fopen(filename, "w"); // Open the output file in write mode
   if (!file)
   {
      fprintf(stderr, "Failed to open output file.\n");
      exit(1);
   }

   int proximityFound = 0;

   for (int i = 0; i < tCount; i++)
   {
      int count = 0;
      int pointIDs[3] = {-1, -1, -1};

      for (int j = 0; j < CONSTRAINTS; j++)
      {
         int pointID = results[i * CONSTRAINTS + j]; // Get the point ID from the results array
         if (pointID >= 0 && pointID < N) // Check if the point ID is valid
         {
            pointIDs[count] = pointID; // Store the point ID in the pointIDs array
            count++;
         }
      }

      if (count == 3) // Check if three points satisfy the proximity criteria
      {
         proximityFound = 1;
         fprintf(file, "Points ");
         for (int j = 0; j < 3; j++)
         {
            fprintf(file, "pointID%d", pointIDs[j]); // Print the point IDs that satisfy the proximity criteria
            if (j < 2)
               fprintf(file, ", ");
         }
         fprintf(file, " satisfy Proximity Criteria at t = %.2f\n", 2.0 * i / tCount - 1); // Print the t value
      }
   }

   if (!proximityFound)
   {
      fprintf(file, "There were no 3 points found for any t.\n"); // Print a message if no points satisfy the proximity criteria
   }

   fclose(file); // Close the file
}

/**
 * Free allocated memory for points, tValues, and results arrays.
 *
 * @param points    Array of points
 * @param tValues   Array of t values
 * @param results   Results array
 */
void freeMemory(Point *points, double *tValues, int *results)
{
   free(points); // Free memory allocated for points array
   free(tValues); // Free memory allocated for tValues array
   free(results); // Free memory allocated for results array
}

/**
 * Main function
 */
int main()
{
   int N, K, tCount;
   double D;
   Point *points;
   double *tValues;
   int *results;

   readInputFile("input.txt", &N, &K, &D, &tCount, &points); // Read input file and populate variables
   calculateTValues(tCount, &tValues); // Calculate t values

   // Initialize results array
   results = (int *)malloc(tCount * CONSTRAINTS * sizeof(int)); // Allocate memory for results array
   if (!results)
   {
      fprintf(stderr, "Failed to allocate results array.\n");
      exit(1);
   }
   memset(results, -1, tCount * CONSTRAINTS * sizeof(int)); // Initialize results array with -1 values

   checkProximityCriteria(points, N, tValues, tCount, D, results, K); // Check proximity criteria between points

   writeOutputFile("output_test.txt", tCount, results, points, N); // Write output file with points that satisfy the proximity criteria

   freeMemory(points, tValues, results); // Free allocated memory

   return 0; // Exit the program
}
