#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_POINTS 1000

typedef struct {
    int id;
    double x1, x2, a, b;
} Point;

double calculate_x(double x1, double x2, double t) {
    return ((x2 - x1) / 2) * sin(t * M_PI / 2) + (x2 + x1) / 2;
}

int satisfies_proximity_criteria(Point* points, int n, int k, double d, double t) {
    int count;
    int i, j;
    double x, y, distance;

    for (i = 0; i < n; i++) {
        count = 0;
        x = calculate_x(points[i].x1, points[i].x2, t);
        y = points[i].a * x + points[i].b;

        for (j = 0; j < n; j++) {
            if (i != j) {
                double x_j = calculate_x(points[j].x1, points[j].x2, t);
                double y_j = points[j].a * x_j + points[j].b;
                distance = sqrt(pow(x - x_j, 2) + pow(y - y_j, 2));

                if (distance < d) {
                    count++;
                    if (count >= k - 1) {
                        printf("count = %d || t = %.3lf || with point %d\n",
                        count, t,points[i].id);
                        return 1;
                    }
                }
            }
        }
    }

    return 0;
}

int main() {
    FILE* input_file = fopen("input.txt", "r");
    FILE* output_file = fopen("output.txt", "w");

    int n, k, t_count;
    double d;
    fscanf(input_file, "%d %d %lf %d", &n, &k, &d, &t_count);

    Point points[MAX_POINTS];
    int i;
    for (i = 0; i < n; i++) {
        fscanf(input_file, "%d %lf %lf %lf %lf", &points[i].id, &points[i].x1, &points[i].x2, &points[i].a, &points[i].b);
    }

    int points_found = 0;
    for (i = 0; i <= t_count; i++) {
        double t = 2 * i / (double)t_count - 1;
        if (satisfies_proximity_criteria(points, n, k, d, t)) {
            fprintf(output_file, "Points ");
            int j;
            for (j = 0; j < n; j++) {
                double x = calculate_x(points[j].x1, points[j].x2, t);
                double y = points[j].a * x + points[j].b;
                fprintf(output_file, "%d ", points[j].id);
                if (j < n - 1) {
                    fprintf(output_file, ",");
                }
            }
            fprintf(output_file, "satisfy Proximity Criteria at t = %.2f\n", t);
            points_found = 1;
            break;
        }
    }

    if (!points_found) {
        fprintf(output_file, "There were no 3 points found for any t.\n");
    }

    fclose(input_file);
    fclose(output_file);

    return 0;
}
