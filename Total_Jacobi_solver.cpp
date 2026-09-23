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
    //Total Jacobi Algorithm for a matrix A

    //Set size and signature
    int size = 6;
    std::vector<double> sig = {-1, 2, -1};
    //Initialise tridiagonal matrix A
    Eigen::MatrixXd A = TriDiagonalMatrix(size, sig, false);
    

    //Initialise R, initialise empty matrices for eigenstuff
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(size, size);
    Eigen::VectorXd eigenvals = Eigen::VectorXd::Zero(size);
    Eigen::MatrixXd eigenvecs = Eigen::MatrixXd::Zero(size, size);

    //Define tolerance and maximum number of iterations
    double tolerance = 1e-8;
    int maxiter = 10000;
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

    //Print outputs for now, not practical on large matrix
    std::cout << "Jacobi eigenvalues: \n";
    for (const auto& element : eigenvals) 
        {
            std::cout << element << " ";
        }
        std::cout << "\n";
    std::cout << "Jacobi eigenvectors: \n";
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << eigenvecs(i,j) << " ";
        }
        std::cout << "\n";
    }

    //Compare to analytical solution of eigenvalues/eigenvectors
    //Copied from my solution for Problem 2//////////////////////////////////////////////////////////////////////////////
    //Find eigenvals/eigenvecs using analytical approach
    Eigen::VectorXd eigenvalues = Eigen::VectorXd::Zero(size);
    Eigen::MatrixXd eigenvectors = Eigen::MatrixXd::Zero(size, size);
    float pi = std::acos(-1.0);
    //Code implementation of the analytical solutions given at https://anderkve.github.io/FYS3150/book/projects/project2.html
    for (int j=1; j<(size+1); j++) //Using j for iterations here to stay consistent with the analytical solutions given
    {
        eigenvalues(j-1) = sig[1] + 2*sig[0]*(std::cos((j*pi)/(size+1)));
        
        for (int l=1; l<(size+1); l++)
        {
            eigenvectors(l-1,j-1) = std::sin((j*l*pi)/(size+1));
        }
    }

    //Normalise eigenvectors
    //According to https://libeigen.gitlab.io/eigen/docs-5.0.1/classEigen_1_1EigenSolver.html the ones from the lib are norm by default
    //Normalise analytical ones:
    for (int i=0; i<size; i++)
    {
        eigenvectors.col(i) = eigenvectors.col(i)/vector_length(eigenvectors.col(i));
    }


    //Print to terminal for comparison
    std::cout << "Analytical eigenvalues: \n";
    for (const auto& element : eigenvalues) 
        {
            std::cout << element << " ";
        }
        std::cout << "\n";

    std::cout << "Analytical eigenvectors: \n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << eigenvectors(i,j) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
