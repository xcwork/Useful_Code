
//                                      svd_gold.h

/*

Copyright (c) 2012 Mark Joshi

released under the GPL v 3.0

adapted from code with the following license

Copyright (C) 2003 Neil Firth

This file is part of QuantLib, a free-software/open-source library
for financial quantitative analysts and developers - http://quantlib.org/

QuantLib is free software: you can redistribute it and/or modify it
under the terms of the QuantLib license.  You should have received a
copy of the license along with this program; if not, please email
<quantlib-dev@lists.sf.net>. The license is also available online at
<http://quantlib.org/license.shtml>.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the license for more details.

Adapted from the TNT project
http://math.nist.gov/tnt/download.html

This software was developed at the National Institute of Standards
and Technology (NIST) by employees of the Federal Government in the
course of their official duties. Pursuant to title 17 Section 105
of the United States Code this software is not subject to copyright
protection and is in the public domain. NIST assumes no responsibility
whatsoever for its use by other parties, and makes no guarantees,
expressed or implied, about its quality, reliability, or any other
characteristic.

We would appreciate acknowledgement if the software is incorporated in
redistributable libraries or applications.

*/

/*! \file svd.hpp
\brief singular value decomposition
*/

#ifndef gold_math_svd_h
#define gold_math_svd_h

#include <gold/MatrixFacade.h>
#include <vector>
#include <gold/math/typedefs_math_gold.h>




//#include <boost/numeric/ublas/lu.hpp>
//#include <boost/numeric/ublas/io.hpp>
//#include <boost/numeric/ublas/matrix.hpp>
//using namespace boost::numeric::ublas;
using namespace std;
//! Singular value decomposition
/*! Refer to Golub and Van Loan: Matrix computation,
The Johns Hopkins University Press

Creating an SVD square object creates a bunch of workspace data.
These will be reused. 

If you want to do your own workspace, a static version of the function is provided.

it is only designed for square matrices so m_ should equal n_ . If not use at your own additional risk. 

*/
class SVDSquareMatrix
{
public:

    static void SVDSolveSquareMatrix( MatrixFacade<Real_d>& input, 
        const Real_d* const target, 
        Real_d* solution,
        Real_d* singular,
        MatrixFacade<Real_d>& U,
        MatrixFacade<Real_d>& V,
        Real_d* e,
        Real_d* work,
        int m_,
        int n_,
        Real_d tolerance);



    SVDSquareMatrix(int squareMatrixSize);

    void SVDSquareMatrix::GetDecomp( const MatrixConstFacade<Real_d>& input,std::vector<Real_d>& singular,
        MatrixFacade<Real_d>& U,
        MatrixFacade<Real_d>& V,
        Real_d tolerance);

    
    void SVDSolve(const MatrixConstFacade<Real_d>& input, 
                               const std::vector<Real_d>& target, 
                               std::vector<Real_d>& solution,
                               Real_d tolerance);

    template<class T>
    void SVDSolve(const MatrixConstFacade<T>& input, 
                   const T* const  target, 
                   T* solution,
                    Real_d tolerance);

//template<class S, class T, class R>
//void SVDSolveBoost(const S& input, 
  //                                  const T& target, 
    //                                R& solution);


	Real_d GetConditionOfLastSolve()
	{
		return singular_vec[0]/singular_vec[n_-1];

	}

private:

    int m_, n_;

    std::vector<Real_d> A_vec;
    std::vector<Real_d> U_vec;


    std::vector<Real_d> V_vec;

    std::vector<Real_d> e_vec;
    std::vector<Real_d> work_vec;

    std::vector<Real_d> target_check_vec;
    std::vector<Real_d> singular_vec;


    std::vector<Real_d> target_copy_vec;
    std::vector<Real_d> solution_vec;





};

template<class T>
void SVDSquareMatrix::SVDSolve(const MatrixConstFacade<T>& input, 
                                  const T * const  target, 
                                  T* solution,
                                  Real_d tolerance)
{


    MatrixFacade<Real_d> A_mat(&A_vec[0],m_,n_);

    for (int i=0; i < m_; ++i)
    {
        target_copy_vec[i] =static_cast<Real_d>(target[i]);
        for (int j=0; j < n_; ++j)
            A_mat(i,j) = static_cast<Real_d>(input(i,j));
    }


    MatrixFacade<Real_d> U_mat(&U_vec[0],m_,n_);
    MatrixFacade<Real_d> V_mat(&V_vec[0],n_,m_);

    SVDSolveSquareMatrix( A_mat, 
        &target_copy_vec[0], 
        &solution_vec[0],
        &singular_vec[0],
        U_mat,
        V_mat,
        &e_vec[0],  
        &work_vec[0], 
        m_, n_, tolerance);

    for (int j=0; j < n_; ++j)
        solution[j] = static_cast<T>(solution_vec[j]);


}
/*
template<class T>
bool InvertMatrix(const matrix<T>& input, matrix<T>& inverse)
{
    typedef permutation_matrix<std::size_t> pmatrix;

    // create a working copy of the input
    matrix<T> A(input);

    // create a permutation matrix for the LU-factorization
    pmatrix pm(A.size1());

    // perform LU-factorization
    int res = lu_factorize(A, pm);
    if (res != 0)
        return false;

    // create identity matrix of "inverse"
    inverse.assign(identity_matrix<T> (A.size1()));

    // backsubstitute to get the inverse
    lu_substitute(A, pm, inverse);

    return true;
}*/

/*
template<class S, class T, class R>
void SVDSquareMatrix::SVDSolveBoost(const S& input, 
                                    const T& target, 
                                    R& solution)
{
  boost::numeric::ublas::matrix<double> A(n_, n_), 
                                        Z(n_, n_);

    for (int i=0; i < n_; ++i)
        for (int j=0; j < n_; ++j)
            A(i,j) = input(i,j);

    InvertMatrix(A, Z);



    if (solution.size() != input.columns())
        GenerateError("missized solution vector in SVDSquareMatrix::SVDSolve");

    if (input.rows() != target.size())
        GenerateError("target and matrix size mismatch in SVDSquareMatrix::SVDSolve");

    for (int i=0; i < n_; ++i)
    {
        Real_d val = 0.0;
        for (int j=0; j < n_; ++j)
            val+= Z(i,j)*target[j];
        solution[i] = val;

    }


#ifdef _DEBUG

    double tolerance = 1E-8;

    bool dumpMatrix = true;

    for (int i=0; i < m_; ++i)
    {
        target_check_vec[i] =0.0;

        for (int j=0; j < n_; ++j)
        {
            //          std::cout << input(i,j) << ",";
            target_check_vec[i] += input(i,j)*solution[j];
        }

        //       std::cout << "\n";

        if (fabs(target_check_vec[i] - target[i])  > tolerance || fp_isnan(solution[i]) )
        {
            std::cout << "error in SVD solve , " << solution[i] << "," << target_check_vec[i] <<" ," << target[i] <<" ,"<< i <<"\n";
            dumpMatrix = true;
        }
    }

    if (dumpMatrix)
    {
        for (int i=0; i < m_; ++i)
        {
            for (int j=0; j < n_; ++j)
                std::cout << input(i,j) << ",";
            std::cout << "\n";

        }
        std::cout << "\n";
        for (int i=0; i < m_; ++i)
            std::cout << singular_vec[i] << "," <<"\n";

    }


#endif

}
*/


#endif

