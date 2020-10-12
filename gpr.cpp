#include <iostream>
#include <vector>
#include <time.h>
#include <omp.h>
using namespace std;

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
        for (size_t j = 0; j < matrix.size(); j++)
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
    vector<double> f;//observed data vector f
    cout << "Size of the Grid is:" << m << "*" << m << endl;
    cout << "Given point is:(" << Rx << ", " << Ry << ")" << endl;
    f = generate_random_array(n);
    print_array(f);
    return 0;
}
