#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <omp.h>
using namespace std;

vector<vector<double>> init_grid_points(int m)
{
    double h = (double)1/(m + 1);
    vector<vector<double>> points;
    for (int i = 1; i <= m; i ++)
    {
        for (int j = 1; j <= m; j ++)
        {
            vector<double> point;
            point.push_back(i*h);
            point.push_back(j*h);
            points.push_back(point);
        }
    }
    return points;
}

vector<double> generate_zero_array(int size)
{
    vector<double> array;
    for (int i = 0; i < size; i++)
    {
        double d = 0.0;
        array.push_back(d);
    }
    return array;
}

vector<vector<double>> generate_zero_matrix(int x, int y)
{
    vector<vector<double>> matrix;
    for (int i = 0; i < x; i++)
    {
        vector<double> row = generate_zero_array(y);
        matrix.push_back(row);
    }
    return matrix;
}

vector<double> init_observed_data_vector(vector<vector<double>> XY)
{
    vector<double> f;
    f = generate_zero_array(XY.size());
    for (size_t i = 0; i < f.size(); i++)
    {
        f[i] = f[i] + 1.0 + pow(XY[i][0] - 0.5, 2) + pow(XY[i][1] - 0.5, 2) + 0.1 * (drand48() - 0.5);
    }
    return f;
}

double compute_predicted_value(vector<vector<double>> XY, vector<double> f, vector<double> rstar)
{
    int n = XY.size();
    vector<vector<double>> A;
    A = generate_zero_matrix(n, n);
    size_t h, i, j;
    int d, t, m;
    //Initialize K
    for (i = 0; i < n; i ++)
    {
        for (j = 0; j < n; j++)
        {
            d = pow(XY[i][0] - XY[j][0], 2) + pow(XY[i][1] - XY[j][1],2);
            A[i][j] = exp(-d);
        }
    }
    //Compute A = tI+K
    t = 0.01;
    for (i = 0; i < n; i ++)
    {
        A[i][i] += t;
    }
    //Compute LU factorization of tI + K
    for (h = 0; h < n - 1; h ++)
    {
        for (i = h + 1; i < n; i ++)
        {
            m = A[i][h] / A[h][h];
            for (j = h + 1; j < n; j ++)
            {
                A[i][j] -= m * A[h][j];
            }
            A[i][h] = m;
        }
    }
    vector<double> k;
    k = generate_zero_array(n);
    for (i = 0; i < n; i ++)
    {
        d = pow(rstar[0]-XY[i][0], 2) + pow(rstar[1]-XY[i][1], 2);
        k[i] = exp(-d);
    }
    double fstar = 1.0;
    //Compute predicted value fstar at rstar
    return fstar;    
}

void print_array(vector<double> array)
{
    for (size_t i = 0; i < array.size(); i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;
}

void print_matrix(vector<vector<double>> matrix)
{
    for (size_t i = 0; i < matrix.size(); i++)
    {
        cout << endl;
        for (size_t j = 0; j < matrix[0].size(); j++)
        {
            cout << matrix[i][j] << " ";
        }
    }
    cout << endl
         << endl;
}

int main(int argc, char** argv) 
{
    srand(time(0));
    double dtime;

    int m = 4;
    vector<double> rstar;
    if (argc > 3){
        m = stoi(argv[1]);
        rstar.push_back(stod(argv[2]));
        rstar.push_back(stod(argv[3]));
    }else{
        cout << "Please indicate grid size and coordinate of r*" << endl;
        return -1;
    }

    vector<vector<double>> XY; //x and y coordinates of grid points
    vector<double> f;//observed data vector f
    double fstar, start, total_time; 
    cout << "Size of the Grid is:" << m << "*" << m << endl;
    cout << "Given point is:(" << rstar[0] << ", " << rstar[1] << ")" << endl;
    XY = init_grid_points(m);
    print_matrix(XY);
    f = init_observed_data_vector(XY);
    print_array(f);
    start = omp_get_wtime();
    fstar = compute_predicted_value(XY, f, rstar); 
    total_time = omp_get_wtime()-start;
    cout << "Predicted value is " << fstar << endl;
    cout << "time (sec) = " << total_time << endl;
    return 0;
}
