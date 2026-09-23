#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "utils.hpp"
#include "eigen-5.0.1/eigen-5.0.1/Eigen/Dense"
#include "eigen-5.0.1/eigen-5.0.1/Eigen/Eigenvalues"

int main()
{
    //Initialise sizes to test and empty iters variable to store
    std::vector<int> sizes{5, 10, 25, 50, 100, 250, 500};
    //Initialise array for writing out the times
    std::vector<std::vector<double>> iter_vals(sizes.size(), std::vector<double>(2, 0));

    for (int i = 0; i<static_cast<int>(sizes.size()); i++)
    {
        //Do Jacobi presets
        int size = sizes[i];
        std::vector<double> sig = {-1, 2, -1};
        //Initialise tridiagonal matrix A
        Eigen::MatrixXd A = TriDiagonalMatrix(size, sig, false);
        
        //Initialise R, initialise empty matrices for eigenstuff
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(size, size);
        Eigen::VectorXd eigenvals = Eigen::VectorXd::Zero(size);
        Eigen::MatrixXd eigenvecs = Eigen::MatrixXd::Zero(size, size);

        //Define tolerance and maximum number of iterations
        double tolerance = 10e-8;
        int maxiter = 1000000;
        bool converged = false;
        int iterations = 0;

        //Run jacobi eigensolver with given params
        jacobi_eigensolver(A,  R, tolerance, eigenvals, eigenvecs, maxiter, iterations, converged);

        //Print output data
        std::cout << "Iters: " << iterations << std::endl;
        std::cout << "Converged: " << converged << std::endl;

        //write iterations number to variable
        iter_vals[i][0] = size;
        iter_vals[i][1] = iterations;
    }

    //Write sizes and outputs as a file for plotting
    // Open the data file for writing
    std::ofstream outFile("Project2_outputs/Size_and_iterations_tridiagonal.txt");

    for (const auto& row : iter_vals) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) {
                outFile << " "; 
            }
        }
        outFile << "\n"; 
    }
    outFile.close();


    return 0;
}
