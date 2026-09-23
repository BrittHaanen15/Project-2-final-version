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
    //Perform a unit test for my jacobi rotation code, before I write it into a function
    //All formulas based on the lecture notes at https://github.com/anderkve/FYS3150/tree/master/lecture_notes/2026

    int size = 4;
    //Our test matrix
    Eigen::MatrixXd A(size, size);
    A << 1, 0, 0, 0,
        0, 1, -0.7, 0,
        0, -0.7, 1, 0,
        0, 0, 0, 1;

    //Initialise ints
    int k, l;

    // Determine the the max off-diagonal element of a symmetric matrix A
    // - Saves the matrix element indicies to k and l 
    // - Returns absolute value of A(k,l) as the function return value
    //This is according to my code that I tested in UnitTest_FindingLargestOffidaElement.cpp
    double off_dia = max_offdiag_symmetric(A, k, l);

    // Initialise R
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(size, size);

    //One jacobi iter
    double tau = (A(l,l) - A(k,k))/(2*A(k,l));
    std::cout << tau << std::endl;
    double exp = (1-1)/(2*0.7);
    std::cout << "Expected: " << exp << std::endl;

    //Pick the smallest one
    double tantheta = 0;
    if (tau >= 0)
    {
        tantheta = 1/(tau + std::sqrt(1 + tau*tau));
    }
    if (tau < 0)
    {
        tantheta = 1/(tau - std::sqrt(1 + tau*tau));
    }

    std::cout << tantheta << std::endl;
    double exp1 = (1/(0 + std::sqrt(1)));
    std::cout << "Expected: " << exp1 << std::endl;

    //Calculate cos and sin
    double costheta = 1/(std::sqrt(1 + tantheta*tantheta));
    double exp2 = 1/(std::sqrt(2));
    std::cout << costheta << std::endl;
    std::cout << "Expected: " << exp2 << std::endl;

    double sintheta = costheta*tantheta;
    std::cout << sintheta << std::endl;
    std::cout << "Expected: " << exp2 << std::endl;

    //Update elements
    double akk_old = A(k,k);
    double all_old = A(l,l);
    A(k,k) = akk_old*costheta*costheta - 2*A(k,l)*costheta*sintheta + all_old*sintheta*sintheta;
    double exp3 = 0.5 - 0.7 + 0.5;
    std::cout << A(k,k) << std::endl;
    std::cout << "Expected: " << exp3 << std::endl;
    A(l,l) = all_old*costheta*costheta + 2*A(k,l)*costheta*sintheta + akk_old*sintheta*sintheta;
    double exp4 = 0.5 + 0.7 + 0.5;
    std::cout << A(l,l) << std::endl;
    std::cout << "Expected: " << exp4 << std::endl;
    A(k,l) = 0;
    A(l,k) = 0;
    std::cout << A(k,l) << std::endl;
    std::cout << A(l,k) << std::endl;

    //Update all other elements
    for (int i = 0; i<size; i++)
    {
        if (i!=k && i!=l)
        {
            double aik_old = A(i,k);
            double ail_old = A(i,l);

            A(i,k) = aik_old*costheta - ail_old*sintheta;
            A(k,i) = A(i,k);
            A(i,l) = ail_old*costheta + aik_old*sintheta;
            A(l,i) = A(i,l);
        }
    }

    //Print matrix as final check
    std::cout << "Our new A is" << std::endl;
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << A(i,j) << " ";
        }
        std::cout << "\n";
    }

    //Update R
    for (int i = 0; i<size; i++)
    {
        double rik_old = R(i,k);
        double ril_old = R(i,l);

        R(i,k) = rik_old*costheta - ril_old*sintheta;
        R(i,l) = ril_old*costheta + rik_old*sintheta;
    }

    //Print R as check
    std::cout << "Our new R is" << std::endl;
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << R(i,j) << " ";
        }
        std::cout << "\n";
    }


    return 0;
}
