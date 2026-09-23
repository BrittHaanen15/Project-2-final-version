#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <chrono>
#include "utils.hpp"
#include "../include/eigen-5.0.1/eigen-5.0.1/Eigen/Dense"
#include "../include/eigen-5.0.1/eigen-5.0.1/Eigen/Eigenvalues"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Make a vector according to start, stop, step
std::vector<double> make_a_range(double start, double stop, double step, bool verbose)
{
    //Calc number of steps based on start, stop and step size
    int stepsnumber = static_cast<int>((stop-start)/step);

    //Preall vector of this size
    std::vector<double> output(stepsnumber);

    for (int i = 0; i < stepsnumber; i++)
    {
        output[i] = start + i*step;
    }

    //Reality check: print generated vector if needed
    if (verbose == true)
    {
        for (const auto& element : output) 
        {
            std::cout << element << " ";
        }
        std::cout << "\n";
    }
    
   return output;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//Generalised matrix algorithm with signature (-1, 2, -1) for problem set 1
//In future: could be good to generalise further to accept different signatures and boundary values, for now it's okay
std::vector<double> General_Algorithm(double step)
{
    //Generate x values
    std::vector<double> x = make_a_range(step, 1, step, false);

    //Generate a, b, c
    std::vector<int> a(x.size(), -1); 
    std::vector<int> b(x.size(), 2);
    std::vector<int> c(x.size(), -1);  

    //Calculate g
    std::vector<double> g(x.size());
    for (int i = 0; i < static_cast<int>(g.size()); i++)
    {
        g[i] = step*step*(100*exp(-10*x[i]));
    }

    //Initialise zero vectors that we fill on the go
    std::vector<double> b_tilde(x.size(), 0); 
    std::vector<double> g_tilde(x.size(), 0); 
    std::vector<double> v(x.size(), 0); 


    //Forward propagation
    b_tilde[0] = b[0];
    g_tilde[0] = g[0];
    for (int i = 1; i < static_cast<int>(g.size()); i++)
    {
        //According to equations given in Lecture Notes at https://github.com/anderkve/FYS3150/blob/master/lecture_notes/2026/lecture_notes.pdf
        //See also citations in project references
        b_tilde[i] = b[i] - (a[i]/b_tilde[i-1])*c[i-1];
        g_tilde[i] = g[i] - (a[i]/b_tilde[i-1])*g_tilde[i-1];
    }

    //Backward substitution
    v[v.size()-1] = g_tilde[v.size()-1]/b_tilde[v.size()-1];
    for (int i = (v.size()-2); i >= 0; i--)
    {
        //According to equations given in Lecture Notes at https://github.com/anderkve/FYS3150/blob/master/lecture_notes/2026/lecture_notes.pdf
        //See also citations in project references
        v[i] = (g_tilde[i] - c[i]*v[i+1])/b_tilde[i];
    }
    
    return v;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//Special matrix algorithm with signature (-1, 2, -1) for problem set 1
std::vector<double> Special_Algorithm(double step)
{   
   //Generate x values
    std::vector<double> x = make_a_range(step, 1, step, false);

    //Generate a, b, c
    std::vector<int> a(x.size(), -1); 
    std::vector<int> b(x.size(), 2);
    std::vector<int> c(x.size(), -1);  

    //Calculate g
    std::vector<double> g(x.size());
    for (int i = 0; i < static_cast<int>(g.size()); i++)
    {
        g[i] = step*step*(100*exp(-10*x[i]));
    }

    //Initialise zero vectors that we fill on the go
    std::vector<double> b_tilde(x.size(), 0); 
    std::vector<double> g_tilde(x.size(), 0); 
    std::vector<double> v(x.size(), 0); 


    //Forward propagation
    b_tilde[0] = 2;
    g_tilde[0] = g[0];
    for (int i = 1; i < static_cast<int>(g.size()); i++)
    {
        //According to equations given in Lecture Notes at https://github.com/anderkve/FYS3150/blob/master/lecture_notes/2026/lecture_notes.pdf
        //See also citations in project references
        b_tilde[i] = 2 - (1/b_tilde[i-1]);
        g_tilde[i] = g[i] + (g_tilde[i-1]/b_tilde[i-1]);
    }

    //Backward substitution
    v[v.size()-1] = g_tilde[v.size()-1]/b_tilde[v.size()-1];
    for (int i = (v.size()-2); i >= 0; i--)
    {
        //According to equations given in Lecture Notes at https://github.com/anderkve/FYS3150/blob/master/lecture_notes/2026/lecture_notes.pdf
        //See also citations in project references
        v[i] = (g_tilde[i] + v[i+1])/b_tilde[i];
    }

return v;
}


///Make a tridiagonal matrix based on signature and size/////////////////////////////////////////////
Eigen::MatrixXd TriDiagonalMatrix(int size, std::vector<double> signature, bool verbose)
{
    //Initialise matrix with zeros of size that we want
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(size, size);
    
    //Assign elements to the three diagonals
    for (int i=0; i<size; i++)
    {
        A(i,i) = signature[1];
        if (i<size-1)
        {
            A(i+1,i) = signature[0];
        }
        if (i>0)
        {
            A(i-1,i) = signature[2];
        }
    }


    //Optional reality check, printing to terminal
    if (verbose==true)
    {
        for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << A(i,j) << " ";
        }
        std::cout << "\n";
    }
    }

    return A;
    
}


///Calculate length of vector////////////////////////////////////////////////////////////////////////////////////
double vector_length(Eigen::VectorXd vec) 
{
    double componentsum = 0;
    std::cout << vec.size() << std::endl;
    for (int i=0; i < vec.size(); i++)
    {
        componentsum += vec(i)*vec(i);
    }

    return std::sqrt(componentsum);
}



///Find biggest off-diagonal element of symmetric matrix////////////////////////////////////////////////////////
//Using the function setup as suggested on https://anderkve.github.io/FYS3150/book/projects/project2.html
double max_offdiag_symmetric(const Eigen::MatrixXd& m, int& k, int& l)
{
    //Loop through rows to find max
    double baseline = 0;
    for (int i=0; i<m.rows(); i++)
    {
        //Loop through relevant columns. For row i, the relevant columns are j=i+1,...,N
        for (int j=(i+1); j<m.cols(); j++)
        {
            if ((i==0)&&(j==(i+1)))
            {
                baseline = std::abs(m(i, j));
                k = i;
                l = j;
            }
            if (std::abs(m(i,j))>baseline)
            {
                baseline = std::abs(m(i, j));
                k = i;
                l = j;
            }
        }
    }

    //Return abs value of this matrix element
    return std::abs(m(k,l));
}


// Perform a single Jacobi rotation, to "rotate away" the off-diagonal element at A(k,l). Assumes symmetric matrix
void jacobi_rotate(Eigen::MatrixXd& A, Eigen::MatrixXd& R, int k, int l)
{
    //One jacobi iteration

    int size = A.rows();
    //Calculate tau
    double tau = (A(l,l) - A(k,k))/(2*A(k,l));

    //Calculate theta based on tau value
    double tantheta = 0;
    if (tau > 0)
    {
        tantheta = 1/(tau + std::sqrt(1 + tau*tau));
    }
    if (tau < 0)
    {
        tantheta = 1/(tau - std::sqrt(1 + tau*tau));
    }
    if (tau == 0)
    {
        tantheta = 1;
    }

    //Calculate cos and sin
    double costheta = 1/(std::sqrt(1 + tantheta*tantheta));
    double sintheta = costheta*tantheta;

    //Update k and l elements
    double akk_old = A(k,k);
    double all_old = A(l,l);

    A(k,k) = akk_old*costheta*costheta - 2*A(k,l)*costheta*sintheta + all_old*sintheta*sintheta;
    A(l,l) = all_old*costheta*costheta + 2*A(k,l)*costheta*sintheta + akk_old*sintheta*sintheta;
    A(k,l) = 0;
    A(l,k) = 0;

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

    //Update R
    for (int i = 0; i<size; i++)
    {
        double rik_old = R(i,k);
        double ril_old = R(i,l);

        R(i,k) = rik_old*costheta - ril_old*sintheta;
        R(i,l) = ril_old*costheta + rik_old*sintheta;
    }

}


// Jacobi method eigensolver://///////////////////////////////////////////////////////////////////////////////////////////////
void jacobi_eigensolver(Eigen::MatrixXd A, Eigen::MatrixXd& R, double eps, Eigen::VectorXd& eigenvalues, Eigen::MatrixXd& eigenvectors, const int maxiter, int& iterations, bool& converged)
{
    // - Runs jacobi_rotate until max off-diagonal element < eps
    // - Writes the eigenvalues as entries in the vector "eigenvalues"
    // - Writes the eigenvectors as columns in the matrix "eigenvectors"
    //   (The returned eigenvalues and eigenvectors are sorted using the snippet below)
    // - Stops if it the number of iterations reaches "maxiter"
    // - Writes the number of iterations to the integer "iterations"
    // - Sets the bool reference "converged" to true if convergence was reached before hitting maxiter
    int size = A.cols();
    int k, l;

    for (int i=0; i<maxiter; i++)
    {
        iterations += 1;

        // Determine the the max off-diagonal element of a symmetric matrix A
        // - Saves the matrix element indicies to k and l 
        // - Returns absolute value of A(k,l) as the function return value
        //This is according to my code that I tested in UnitTest_FindingLargestOffidaElement.cpp
        double off_dia = max_offdiag_symmetric(A, k, l);

        if (off_dia < eps)
        {
            //If largest off diagonal element is less than tolerance, exit loop and set converge to true
            converged = true;
            break;
        }

        //If largest off diagonal element is larger than tolerance, perform jra
            jacobi_rotate(A, R, k, l);

    }

    //Read eigenvals from A matrix and eigenvectors from R matrix
    for (int i=0; i<size; i++)
    {
        eigenvalues(i) = A(i,i);
        eigenvectors.col(i) = R.col(i)/R.col(i).norm(); //Normalise eigenvecs as I write them
    }
}
