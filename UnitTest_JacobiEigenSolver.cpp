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
    //Perform a unit test for my total jacobi rotation solvercode, before I write it into a function
    //All formulas based on the lecture notes at https://github.com/anderkve/FYS3150/tree/master/lecture_notes/2026

    int size = 4;
    //Our test matrix
    Eigen::MatrixXd A(size, size);
    A << 1, 0, 0, 0,
        0, 1, -0.7, 0,
        0, -0.7, 1, 0,
        0, 0, 0, 1;

    //Initialise ints k and l, initialise R
    int k, l;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(size, size);


    // Performs a single Jacobi rotation, to "rotate away"
    // the off-diagonal element at A(k,l).
    // - Assumes symmetric matrix, so we only consider k < l
    // - Modifies the input matrices A and R
    jacobi_rotate(A, R, k, l);

    //Define tolerance and maximum number of iterations
    double tolerance = 10e-8;
    int maxiter = 10000;
    bool converged = false;
    int iterations = 0;

    for (int i=0; i<maxiter; i++)
    {
        iterations += 1;

        // Determine the the max off-diagonal element of a symmetric matrix A
        // - Saves the matrix element indicies to k and l 
        // - Returns absolute value of A(k,l) as the function return value
        //This is according to my code that I tested in UnitTest_FindingLargestOffidaElement.cpp
        double off_dia = max_offdiag_symmetric(A, k, l);

        if (off_dia > tolerance)
        {
            //If largest off diagonal element is larger than tolerance, perform jra
            jacobi_rotate(A, R, k, l);
        }
        if (off_dia < tolerance)
        {
            //If largest off diagonal element is less than tolerance, exit loop and set converge to true
            converged = true;
            break;
        }

    }

    //Read eigenvals from A matrix and eigenvectors from R matrix
    Eigen::VectorXd eigenvals = Eigen::VectorXd::Zero(size);
    Eigen::MatrixXd eigenvecs = Eigen::MatrixXd::Zero(size, size);

    for (int i=0; i<size; i++)
    {
        eigenvals(i) = A(i,i);
        eigenvecs.col(i) = R.col(i)/R.col(i).norm(); //Normalise eigenvecs as I write them
    }

    //Print outputs as a check
    std::cout << "Iters: " << iterations << std::endl;
    std::cout << "Converged: " << converged << std::endl;
    for (const auto& element : eigenvals) 
        {
            std::cout << element << " ";
        }
        std::cout << "\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << eigenvecs(i,j) << " ";
        }
        std::cout << "\n";
    }
    
    return 0;
}

