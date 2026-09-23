#include <iostream>
#include <vector>
#include <numeric>   // for std::iota
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
    //Initialise x_hat vector
    double h = 0.1;
    std::vector<double> x_hat = make_a_range(h, 1, h, false);
    int size = (1/h) - 1; //Since N=n-1 and h=1/n
    std::cout << "\n"; // Print a new line at the end

    //Initialise tridiagonal matrix A
    std::vector<double> sig = {(-1/(h*h)), 2/(h*h), (-1/(h*h))};
    Eigen::MatrixXd A = TriDiagonalMatrix(size, sig, false);
        
    //Initialise R, initialise empty matrices for eigenstuff
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(size, size);
    Eigen::VectorXd eigenvals = Eigen::VectorXd::Zero(size);
    Eigen::MatrixXd eigenvecs = Eigen::MatrixXd::Zero(size, size);

    //Define tolerance and maximum number of iterations
    double tolerance = 1e-8;
    int maxiter = 1000000;
    bool converged = false;
    int iterations = 0;

    //Run jacobi eigensolver with given params
    jacobi_eigensolver(A,  R, tolerance, eigenvals, eigenvecs, maxiter, iterations, converged);

    //Print output data
    std::cout << "Iters: " << iterations << std::endl;
    std::cout << "Converged: " << converged << std::endl;

    // Helper snippet to sort eigenvalues and eigenvectors at the end of jacobi_eigensolver
    // As given at https://anderkve.github.io/FYS3150/book/projects/project2.html#code-snippets
    // Helper snippet to sort eigenvalues and eigenvectors at the end of jacobi_eigensolver:
    // 1. Get index ordering that sorts eigenvalues
    std::vector<int> idx(eigenvals.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return eigenvals(i) < eigenvals(j); });

    // 2. Sort eigenvalues and columns of eigenvectors
    // .eval() is needed so we don't overwrite values before they have been copied!
    eigenvals = eigenvals(idx).eval();
    eigenvecs = eigenvecs(Eigen::placeholders::all, idx).eval();

    //Write first 3 eigenvectors as outputs for plotting
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(size + 2, 4);
    output.block(1, 0, x_hat.size(), 1) = Eigen::Map<Eigen::VectorXd>(x_hat.data(), x_hat.size());; //write x hat to first col
    output(output.rows()-1, 0) = 1; //set boundary vals
    
     for (int i=1; i<4; i++)
     {
         output.block(1, i, eigenvecs.rows(), 1) = eigenvecs.col(i-1);
     }

    // Open the data file for writing
    std::ofstream outFile("Project2_outputs/Discrete_solution_n10.txt");

    // 3. Write the matrix to the file
    if (outFile.is_open()) {
        outFile << output << "\n";
        outFile.close();
    } 
    outFile.close();


    return 0;
}
