#include <iostream>
#include <vector>
#include <time.h>
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

vector<double> generate_random_array(int size)
{
    vector<double> array;
    for (int i = 0; i < size; i++)
    {
        double d = 0.1 * (drand48() - 0.5);
        array.push_back(d);
    }
    return array;
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
    double Rx = 0.5, Ry = 0.5;
    if (argc > 3){
        m = stoi(argv[1]);
        Rx = stod(argv[2]);
        Ry = stod(argv[3]);
    }else{
        cout << "Please indicate grid size and coordinate of r*" << endl;
        return -1;
    }

    int n = m * m;
    vector<vector<double>> XY; //x and y coordinates of grid points
    vector<double> f;//observed data vector f
    
    cout << "Size of the Grid is:" << m << "*" << m << endl;
    cout << "Given point is:(" << Rx << ", " << Ry << ")" << endl;
    XY = init_grid_points(m);
    print_matrix(XY);
    f = generate_random_array(n);
    print_array(f);
    return 0;
}
