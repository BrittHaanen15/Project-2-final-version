#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <chrono>
#include "utils.hpp"
#include "eigen-5.0.1/eigen-5.0.1/Eigen/Dense"
#include "eigen-5.0.1/eigen-5.0.1/Eigen/Eigenvalues"

int main()
{
    //Set size and signature
    int size = 6;
    std::vector<double> sig = {-1, 2, -1};

    //Initialise matrix with zeros 
    Eigen::MatrixXd Matrix = TriDiagonalMatrix(size, sig, false);

    //Find eigenvals/eigenvecs using Eigen
    //Syntax for this + printing at the end from https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1SelfAdjointEigenSolver.html
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es; //Initialise eigensolver thingy
    es.compute(Matrix);

    //Find eigenvals/eigenvecs using analytical approach
    Eigen::VectorXd eigenvals = Eigen::VectorXd::Zero(size);
    Eigen::MatrixXd eigenvecs = Eigen::MatrixXd::Zero(size, size);
    float pi = std::acos(-1.0);
    //Code implementation of the analytical solutions given at https://anderkve.github.io/FYS3150/book/projects/project2.html
    for (int j=1; j<(size+1); j++) //Using j for iterations here to stay consistent with the analytical solutions given
    {
        eigenvals(j-1) = sig[1] + 2*sig[0]*(std::cos((j*pi)/(size+1)));
        
        for (int l=1; l<(size+1); l++)
        {
            eigenvecs(l-1,j-1) = std::sin((j*l*pi)/(size+1));
        }
    }

    //Normalise eigenvectors
    //According to https://libeigen.gitlab.io/eigen/docs-5.0.1/classEigen_1_1EigenSolver.html the ones from the lib are norm by default
    //Normalise analytical ones:
    for (int i=0; i<size; i++)
    {
        eigenvecs.col(i) = eigenvecs.col(i)/vector_length(eigenvecs.col(i));
    }


    //Print both eigenvals to terminal for comparison
    std::cout << "Eigenvalues (Eigen): " << es.eigenvalues().transpose() << std::endl;
    std::cout << "Analytical result: \n";
    for (const auto& element : eigenvals) 
        {
            std::cout << element << " ";
        }
        std::cout << "\n";

    //Print both eigenvecs to terminal for comparison
    std::cout << "Eigenvectors (Eigen): " << std::endl << es.eigenvectors() << std::endl << std::endl;
    std::cout << "Analytical result: \n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << eigenvecs(i,j) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
