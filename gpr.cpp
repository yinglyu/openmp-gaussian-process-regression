#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include <omp.h>
using namespace std;

void print_array(vector<double> &array);
void print_matrix(vector<vector<double>> &matrix);

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

vector<double> init_observed_data_vector(vector<vector<double>> XY)
{
    vector<double> f(XY.size(), 0);
    # pragma omp parallel default(shared)
    # pragma omp for
    for (size_t i = 0; i < f.size(); i++)
    {
        f[i] = 1.0 + pow(XY[i][0] - 0.5, 2) + pow(XY[i][1] - 0.5, 2) + 0.1 * (drand48() - 0.5);
    }
    return f;
}

double compute_predicted_value(vector<vector<double>> &XY, vector<double> &f, vector<double> &rstar)
{
    int n = XY.size();
    vector<vector<double>> A(n, vector<double>(n, 0));
    int h, i, j;
    double d, t, m;
    //Initialize K
    # pragma omp parallel for collapse(2) shared(A, XY) private(i, j)
    for (i = 0; i < n; i ++)
    {
        for (j = 0; j < n; j++)
        {
            d = pow(XY[i][0] - XY[j][0], 2) + pow(XY[i][1] - XY[j][1],2);
            A[i][j] = exp(-d);
        }
    }
    //cout << "K:" << endl;
    //print_matrix(A);
    //Compute A = tI+K
    t = 0.01;
    # pragma omp parallel for shared(A) private(i)
    for (i = 0; i < n; i ++)
    {
        A[i][i] += t;
    }
    //cout << "A:" << endl;
    //print_matrix(A);
    //Compute LU factorization of tI + K
    for (h = 0; h < n - 1; h ++)
    {
        for (i = h + 1; i < n; i ++)
        {
            m = A[i][h] / A[h][h];
            # pragma omp parallel for shared(A) private(j)
            for (j = h + 1; j < n; j ++)
            {
                A[i][j] -= m * A[h][j];
            }
            A[i][h] = m;
        }
    }
    //cout << "LU:" << endl;
    //print_matrix(A);
    vector<double> k(n, 0);
    # pragma omp parallel for default(shared) private(i, d)
    for (i = 0; i < n; i ++)
    {
        d = pow(rstar[0]-XY[i][0], 2) + pow(rstar[1]-XY[i][1], 2);
        k[i] = exp(-d);
    }
    //cout << "k:" << endl;
    //print_array(k);
    
    //Solve Az = f LUz = f
    //1. Solve Ly = f for y
    vector<double> y(n, 0);
    for (i = 0; i < n; i ++)
    {
        m = 0;
        # pragma omp parallel for private(j) reduction(+:m)
        for (j = 0; j < i; j ++)
        {
            m += A[i][j] * y[j];
        }
        y[i] = f[i] - m;
    }
    //cout << "y:" << endl;
    //print_array(y);
    //2. Solve Uz = y for z
    vector<double> z(n, 0);
    for (i = n - 1; i >= 0; i --)
    {
        m = 0;
        # pragma omp parallel for private(j) reduction(+:m)
        for (j = i + 1; j < n; j ++)
        {
            m += A[i][j] * z[j];
        }
        z[i] = (y[i]-m)/A[i][i];
    }
    //cout << "z:" << endl; 
    //print_array(z);
    double fstar = 0.0;
    //Compute predicted value fstar at rstar: k'*z
    # pragma omp parallel for private(i) reduction(+:fstar)
    for (i = 0; i < n; i ++)
    {
        fstar += k[i] * z[i];
    }
    return fstar;    
}

void print_array(vector<double> &array)
{
    for (size_t i = 0; i < array.size(); i++)
    {
        cout << array[i] << " ";
    }
    cout << endl;
}

void print_matrix(vector<vector<double>> &matrix)
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
    XY = init_grid_points(m);
    //print_matrix(XY);
    
    f = init_observed_data_vector(XY);
    //print_array(f);
    
    start = omp_get_wtime();
    fstar = compute_predicted_value(XY, f, rstar); 
    total_time = omp_get_wtime()-start;
    
    int p;
    #pragma omp parallel
    {
        p = omp_get_num_threads();
    }
    
    cout << "m = " << m;
    cout << ", p = " << p;
    cout << ", f(" << rstar[0] << ", " << rstar[1] << ") = " << fstar;
    cout << ", time (sec) = " << total_time << endl;
    return 0;
}
