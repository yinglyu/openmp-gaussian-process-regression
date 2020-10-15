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

vector<vector<double>> compute_A(vector<vector<double>> &XY)
{
    int n = XY.size();
    vector<vector<double>> A(n, vector<double>(n, 0));
    int i, j;
    double d, t;
    //Initialize K
    # pragma omp parallel proc_bind(close) shared(XY, A) private(i, j, d)
    {
        # pragma omp for collapse(2)
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
        # pragma omp for
        for (i = 0; i < n; i ++)
        {
            A[i][i] += t;
        }
    }
    return A;
}

vector<vector<double>> compute_LU_factors(vector<vector<double>> A)
{
    int n = A.size();
    int k, i, j;
    double m;
    for (k = 0; k < n - 1; k ++)
    {
        # pragma omp parallel for shared(A) private(i, j, m) proc_bind(close) 
        for (i = k + 1; i < n; i ++)
        {
            m = A[i][k] / A[k][k];
            for (j = k + 1; j < n; j ++)
            {
                A[i][j] -= m * A[k][j];
            }
            A[i][k] = m;
        }
    }
    return A;
}

vector<double> solve_triangular_systems(vector<vector<double>> A, vector<double> f)
{
    int n = A.size();
    vector<double> y(n, 0);
    vector<double> z(n, 0);
    int i, j;
    double m; 
    //Solve Az = f by LUz = f
    //1. Solve Ly = f for y
    for (i = 0; i < n; i ++)
    {
        m = 0;
        int chunk = i/omp_get_num_threads();
        # pragma omp parallel default(shared) proc_bind(close)
        {
        # pragma omp for private(j) reduction(+:m) schedule(static, chunk)
        for (j = 0; j < i; j ++)
        {
            m += A[i][j] * y[j];
        }
        }

        y[i] = f[i] - m;
    }
    //cout << "y:" << endl;
    //print_array(y);
    
    //2. Solve Uz = y for z
    for (i = n - 1; i >= 0; i --)
    {
        m = 0;
        int chunk = (n-i)/omp_get_num_threads();
        # pragma omp parallel default(shared) proc_bind(close)
        {
        # pragma omp for private(j) reduction(+:m) schedule(static, chunk)
        for (j = i + 1; j < n; j ++)
        {
            m += A[i][j] * z[j];
        }
        }
        z[i] = (y[i]-m)/A[i][i];
        
    }
    //cout << "z:" << endl; 
    //print_array(z);   
    return z;
}

vector<double> compute_k(vector<vector<double>> &XY, vector<double> &rstar)
{
    int i, n = XY.size(); 
    vector<double> k(n, 0);
    double d;
    # pragma omp parallel for default(shared) private(i, d)
    for (i = 0; i < n; i ++)
    {
        d = pow(rstar[0]-XY[i][0], 2) + pow(rstar[1]-XY[i][1], 2);
        k[i] = exp(-d);
    }
    //cout << "k:" << endl;
    //print_array(k);
    return k; 
}

double compute_fstar(vector<double> &k, vector<double> &z)
{
    size_t i, n = k.size();
    double fstar = 0.0;
    // Compute predicted value fstar at rstar: k'*z
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

    vector<vector<double>> XY, A, LU;     
    vector<double> f, k, z;    
    double fstar, start, LU_time, solver_time, LU_floats, solver_floats; 
    
    XY = init_grid_points(m);//x and y coordinates of grid points
    //cout << "XY:" << endl;
    //print_matrix(XY);
    
    f = init_observed_data_vector(XY);//observed data vector f
    //cout << "f:" << endl;
    //print_array(f);
    
    A = compute_A(XY);//tI+K
    //cout << "A:" << endl;
    //print_matrix(A);
    
    k = compute_k(XY, rstar);
    //cout << "k:" << endl;
    //print_array(k); 
    
    double n = XY.size();
    LU_floats = n*(n-1)*(4*n+1)/6;
    solver_floats = n*(4+n);
    
    vector<int> threads = {1, 2, 4, 8, 16, 20};
    for (int i = 0; i < threads.size(); i++)
    {
        omp_set_num_threads(threads[i]);
        
        start = omp_get_wtime();
        LU = compute_LU_factors(A); //LU factorization of A
        //cout << "LU:" <<endl;
        //print_matrix(LU);
        LU_time = omp_get_wtime()-start;

        start = omp_get_wtime(); 
        z = solve_triangular_systems(LU, f);
        //cout << "z:" << endl;
        //print_array(z); 
        solver_time = omp_get_wtime()-start;
        
        fstar = compute_fstar(k, z);
        int p;
        #pragma omp parallel
        {
            p = omp_get_num_threads();
        }
        
        cout << "m = " << m;
        cout << ", p = " << p;
        cout << ", f(" << rstar[0] << ", " << rstar[1] << ") = " << fstar;
        cout << ", LU_time (sec) = " << LU_time;
        cout << ", LU_FLOPS = " << LU_floats/LU_time;
        cout << ", solver_time (sec) = " << solver_time;
        cout << ", solver_FLOPS = " << solver_floats/solver_time;
         
        cout << endl;
    }
    return 0;
}
