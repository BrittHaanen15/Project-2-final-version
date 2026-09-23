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
    //Our test matrix
    Eigen::MatrixXd A(4,4);
    A << 1, 0, 0, 0.5,
        0, 1, -0.7, 0,
        0, -0.7, 1, 0,
        0.5, 0, 0, 1;

    //Initialise ints
    int k, l;

    double off_dia = max_offdiag_symmetric(A, k, l);

    //Print result
    std::cout << "Remember that these will be 0 indexed, while lec notes assume 1 indexed!" << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "l: " << l << std::endl;
    std::cout << "Element value: " << off_dia << std::endl;

    //As extra test, set element that was found to zero and try to find new!
    A(k,l) = 0;
    off_dia = max_offdiag_symmetric(A, k, l);

    //Print result
    std::cout << "Second iter:" << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "l: " << l << std::endl;
    std::cout << "Element value: " << off_dia << std::endl;

    //Final test, set also this element to zero, it should remain stuck at k=0 and l=1 this time. Value zero
    A(k,l) = 0;
    off_dia = max_offdiag_symmetric(A, k, l);

    //Print result
    std::cout << "Third iter:" << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "l: " << l << std::endl;
    std::cout << "Element value: " << off_dia << std::endl;

    return 0;
}
