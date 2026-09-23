//Make a vector according to start, stop, step
std::vector<double> make_a_range(double start, double stop, double step, bool verbose);

//Generalised matrix algorithm with signature (-1, 2, -1) for problem set 1
//In future: could be good to generalise further to accept different signatures and boundary values, for now it's okay
std::vector<double> General_Algorithm(double step);

//Special matrix algorithm with signature (-1, 2, -1) for problem set 1
std::vector<double> Special_Algorithm(double step);

//Make a tridiagonal matrix according to signature and size, specify if you want it printed to terminal
#include "eigen-5.0.1/eigen-5.0.1/Eigen/Dense"
Eigen::MatrixXd TriDiagonalMatrix(int size, std::vector<double> signature, bool verbose);

///Calculate length of vector
double vector_length(Eigen::VectorXd vec);

///Find biggest off-diagonal element of symmetric matrix
//Using the function setup as suggested on https://anderkve.github.io/FYS3150/book/projects/project2.html
double max_offdiag_symmetric(const Eigen::MatrixXd& m, int& k, int& l);

// Perform a single Jacobi rotation, to "rotate away" the off-diagonal element at A(k,l). Assumes symmetric matrix
void jacobi_rotate(Eigen::MatrixXd& A, Eigen::MatrixXd& R, int k, int l);

//Jacobi method eigensolver: Runs jacobi_rotate until max off-diagonal element < eps
void jacobi_eigensolver(Eigen::MatrixXd A, Eigen::MatrixXd& R, double eps, Eigen::VectorXd& eigenvalues, Eigen::MatrixXd& eigenvectors, const int maxiter, int& iterations, bool& converged);