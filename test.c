#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "myProto.h"

void readInputFile(const char *filename, int *N, int *K, double *D, int *tCount, Point **points)
{
   FILE *file = fopen(filename, "r");
   if (!file)
   {
      fprintf(stderr, "Failed to open input file.\n");
      exit(1);
   }

   if (fscanf(file, "%d %d %lf %d\n", N, K, D, tCount) != 4)
   {
      fprintf(stderr, "Failed to read required values from input file.\n");
      fclose(file);
      exit(1);
   }

   *points = (Point *)malloc(*N * sizeof(Point));
   if (!(*points))
   {
      fprintf(stderr, "Failed to allocate points.\n");
      fclose(file);
      exit(1);
   }

   for (int i = 0; i < *N; ++i)
   {
      if (fscanf(file, "%d %lf %lf %lf %lf\n", &((*points)[i].id), &((*points)[i].x1), &((*points)[i].x2), &((*points)[i].a), &((*points)[i].b)) != 5)
      {
         fprintf(stderr, "Failed to read point data from input file.\n");
         fclose(file);
         exit(1);
      }
   }

   fclose(file);
}

void calculateTValues(int tCount, double **tValues)
{
   *tValues = (double *)malloc((tCount + 1) * sizeof(double));

   // calculate all t points
   for (int i = 0; i <= tCount; ++i)
   {
      (*tValues)[i] = (2.0 * i / tCount) - 1;
   }
}

double calcDistance(const Point *p1, const Point *p2, double *t)
{
   double x1 = ((p1->x2 - p1->x1) / 2) * sin((*t) * M_PI / 2) + ((p1->x2 + p1->x1) / 2);
   double y1 = p1->a * x1 + p1->b;

   double x2 = ((p2->x2 - p2->x1) / 2) * sin((*t) * M_PI / 2) + ((p2->x2 + p2->x1) / 2);
   double y2 = p2->a * x2 + p2->b;

   double distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));

   return distance;
}

int isProximityCriteriaMet(const Point *p1, const Point *p2, double *t, double D)
{
   double distance = calcDistance(p1, p2, t);
   return distance <= D;
}

void updateResults(int idx, int *results, int proximityPointId)
{
   for (int j = 0; j < CONSTRAINTS; j++)
   {
      int targetIndex = idx * CONSTRAINTS + j;
      if (results[targetIndex] == -1)
      {
         results[targetIndex] = proximityPointId;
         return;
      }
   }
}

void checkProximityCriteria(Point *points, int N, double *tValues, int tCount, double D, int *results)
{
   for (int i = 0; i < tCount; i++)
   {
      for (int j = 0; j < N - 1; j++)
      {
         for (int k = j + 1; k < N; k++)
         {
            if (isProximityCriteriaMet(&points[j], &points[k], &tValues[i], D))
            {
               updateResults(i, results, points[j].id);
               updateResults(i, results, points[k].id);
            }
         }
      }
   }
}

void writeOutputFile(const char *filename, int tCount, int *results, Point *points, int N)
{
   FILE *file = fopen(filename, "w");
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

   if (!proximityFound)
   {
      fprintf(file, "There were no 3 points found for any t.\n");
   }

   fclose(file);
}

void freeMemory(Point *points, double *tValues, int *results)
{
   free(points);
   free(tValues);
   free(results);
}

int main()
{
   int N, K, tCount;
   double D;
   Point *points;
   double *tValues;
   int *results;

   readInputFile("input.txt", &N, &K, &D, &tCount, &points);
   calculateTValues(tCount, &tValues);

   // Initialize results array
   results = (int *)malloc(tCount * CONSTRAINTS * sizeof(int));
   memset(results, -1, tCount * CONSTRAINTS * sizeof(int));

   checkProximityCriteria(points, N, tValues, tCount, D, results);

   writeOutputFile("output.txt", tCount, results, points, N);

   freeMemory(points, tValues, results);

   return 0;
}
