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


#define BOOST_UBLAS_NO_EXCEPTIONS
#include <gold/math/svd_gold.h>
#include <gold/math/fp_utilities.h>

#include <cmath>
#include <algorithm>
#include <iostream>


/* Matrix inversion routine.
Uses lu_factorize and lu_substitute in uBLAS to invert a matrix */


namespace {

    /*  returns hypotenenuse of real (non-complex) scalars a and b by
    avoiding underflow/overflow
    using (a * sqrt( 1 + (b/a) * (b/a))), rather than
    sqrt(a*a + b*b).
    */
    Real_d hypoten(const Real_d &a, const Real_d &b) {
        if (a == 0) {
            return std::fabs(b);
        } else {
            Real_d c = b/a;
            return std::fabs(a) * std::sqrt(1 + c*c);
        }
    }

}

SVDSquareMatrix::SVDSquareMatrix(int squareMatrixSize):m_(squareMatrixSize), n_(squareMatrixSize), A_vec(m_*n_),U_vec(m_*n_),V_vec(m_*n_) , e_vec(n_), work_vec(m_), target_check_vec(squareMatrixSize),singular_vec(squareMatrixSize),
           target_copy_vec(squareMatrixSize),
          solution_vec(squareMatrixSize)
{

}

void SVDSquareMatrix::SVDSolve(const MatrixConstFacade<Real_d>& input, 
                               const std::vector<Real_d>& target, 
                               std::vector<Real_d>& solution,
                               Real_d tolerance)
{

    if (solution.size() != input.columns())
        GenerateError("missized solution vector in SVDSquareMatrix::SVDSolve");

    if (input.rows() != target.size())
        GenerateError("target and matrix size mismatch in SVDSquareMatrix::SVDSolve");


    MatrixFacade<Real_d> A_mat(&A_vec[0],m_,n_);

    for (int i=0; i < m_; ++i)
        for (int j=0; j < n_; ++j)
            A_mat(i,j) = input(i,j);

	U_vec.resize(0);
	U_vec.resize(m_*n_,0.0);
	
	V_vec.resize(0);
	V_vec.resize(m_*n_,0.0);
	 e_vec.resize(0);
	 e_vec.resize(n_,0.0);
	 work_vec.resize(0);
	 work_vec.resize(m_,0);

	 singular_vec.resize(0);
	 singular_vec.resize(m_,0);


	  solution.resize(0);
	 solution.resize(m_,0);

    MatrixFacade<Real_d> U_mat(&U_vec[0],m_,n_);
    MatrixFacade<Real_d> V_mat(&V_vec[0],n_,m_);

    SVDSolveSquareMatrix( A_mat, 
        &target[0], 
        &solution[0],
        &singular_vec[0],
        U_mat,
        V_mat,
        &e_vec[0],  
        &work_vec[0], 
        m_, n_, tolerance);


}

void SVDSquareMatrix::GetDecomp(const MatrixConstFacade<Real_d>& input, std::vector<Real_d>& singular_vec,
                                MatrixFacade<Real_d>& U_mat,
                                MatrixFacade<Real_d>& V_mat,
                               Real_d tolerance)
{
    MatrixFacade<Real_d> A_mat(&A_vec[0],m_,n_);

    for (int i=0; i < m_; ++i)
        for (int j=0; j < n_; ++j)
            A_mat(i,j) = input(i,j);

   
    SVDSolveSquareMatrix( A_mat, 
        0,  
        0,
        &singular_vec[0],
        U_mat,
        V_mat,
        &e_vec[0],  
        &work_vec[0], 
        m_, n_, tolerance);

}





void SVDSquareMatrix::SVDSolveSquareMatrix( MatrixFacade<Real_d>& A, 
                                           const Real_d* const target, 
                                           Real_d* solution,
                                           Real_d* singular,
                                           MatrixFacade<Real_d>& U_mat,
                                           MatrixFacade<Real_d>& V_mat,
                                           Real_d* e,
                                           Real_d* work, int m_, int n_,
                                           Real_d tolerance)
{



    Integer i, j, k;

    // Reduce A to bidiagonal form, storing the diagonal elements
    // in s and the super-diagonal elements in e.

    Integer nct = std::min(m_-1,n_);
    Integer nrt = std::max(0,n_-2);
    for (k = 0; k < std::max(nct,nrt); k++) {
        if (k < nct) {

            // Compute the transformation for the k-th column and
            // place the k-th diagonal in s[k].
            // Compute 2-norm of k-th column without under/overflow.
            singular[k] = 0;
            for (i = k; i < m_; i++) {
                singular[k] = hypoten(singular[k],A[i][k]);
            }
            if (singular[k] != 0.0) {
                if (A[k][k] < 0.0) {
                    singular[k] = -singular[k];
                }
                for (i = k; i < m_; i++) {
                    A[i][k] /= singular[k];
                }
                A[k][k] += 1.0;
            }
            singular[k] = -singular[k];
        }
        for (j = k+1; j < n_; j++) {
            if ((k < nct) && (singular[k] != 0.0))  {

                // Apply the transformation.

                Real_d t = 0;
                for (i = k; i < m_; i++) {
                    t += A[i][k]*A[i][j];
                }
                t = -t/A[k][k];
                for (i = k; i < m_; i++) {
                    A[i][j] += t*A[i][k];
                }
            }

            // Place the k-th row of A into e for the
            // subsequent calculation of the row transformation.

            e[j] = A[k][j];
        }
        if (k < nct) {

            // Place the transformation in U for subsequent back
            // multiplication.

            for (i = k; i < m_; i++) {
                U_mat[i][k] = A[i][k];
            }
        }
        if (k < nrt) {

            // Compute the k-th row transformation and place the
            // k-th super-diagonal in e[k].
            // Compute 2-norm without under/overflow.
            e[k] = 0;
            for (i = k+1; i < n_; i++) {
                e[k] = hypoten(e[k],e[i]);
            }
            if (e[k] != 0.0) {
                if (e[k+1] < 0.0) {
                    e[k] = -e[k];
                }
                for (i = k+1; i < n_; i++) {
                    e[i] /= e[k];
                }
                e[k+1] += 1.0;
            }
            e[k] = -e[k];
            if ((k+1 < m_) & (e[k] != 0.0)) {

                // Apply the transformation.

                for (i = k+1; i < m_; i++) {
                    work[i] = 0.0;
                }
                for (j = k+1; j < n_; j++) {
                    for (i = k+1; i < m_; i++) {
                        work[i] += e[j]*A[i][j];
                    }
                }
                for (j = k+1; j < n_; j++) {
                    Real_d t = -e[j]/e[k+1];
                    for (i = k+1; i < m_; i++) {
                        A[i][j] += t*work[i];
                    }
                }
            }

            // Place the transformation in V for subsequent
            // back multiplication.

            for (i = k+1; i < n_; i++) {
                V_mat[i][k] = e[i];
            }
        }
    }

    // Set up the final bidiagonal matrix or order n.

    if (nct < n_) {
        singular[nct] = A[nct][nct];
    }
    if (nrt+1 < n_) {
        e[nrt] = A[nrt][n_-1];
    }
    e[n_-1] = 0.0;

    // generate U

    for (j = nct; j < n_; j++) {
        for (i = 0; i < m_; i++) {
            U_mat[i][j] = 0.0;
        }
        U_mat[j][j] = 1.0;
    }
    for (k = nct-1; k >= 0; --k) {
        if (singular[k] != 0.0) {
            for (j = k+1; j < n_; ++j) {
                Real_d t = 0;
                for (i = k; i < m_; i++) {
                    t += U_mat[i][k]*U_mat[i][j];
                }
                t = -t/U_mat[k][k];
                for (i = k; i < m_; i++) {
                    U_mat[i][j] += t*U_mat[i][k];
                }
            }
            for (i = k; i < m_; i++ ) {
                U_mat[i][k] = -U_mat[i][k];
            }
            U_mat[k][k] = 1.0 + U_mat[k][k];
            for (i = 0; i < k-1; i++) {
                U_mat[i][k] = 0.0;
            }
        } else {
            for (i = 0; i < m_; i++) {
                U_mat[i][k] = 0.0;
            }
            U_mat[k][k] = 1.0;
        }
    }

    // generate V

    for (k = n_-1; k >= 0; --k) {
        if ((k < nrt) & (e[k] != 0.0)) {
            for (j = k+1; j < n_; ++j) {
                Real_d t = 0;
                for (i = k+1; i < n_; i++) {
                    t += V_mat[i][k]*V_mat[i][j];
                }
                t = -t/V_mat[k+1][k];
                for (i = k+1; i < n_; i++) {
                    V_mat[i][j] += t*V_mat[i][k];
                }
            }
        }
        for (i = 0; i < n_; i++) {
            V_mat[i][k] = 0.0;
        }
        V_mat[k][k] = 1.0;
    }

    // Main iteration loop for the singular values.

    Integer p = n_, pp = p-1;
    Integer iter = 0;
    Real_d eps = std::pow(2.0,-52.0);
    while (p > 0) {
        Integer k;
        Integer kase;

        // Here is where a test for too many iterations would go.

        // This section of the program inspects for
        // negligible elements in the s and e arrays.  On
        // completion the variables kase and k are set as follows.

        // kase = 1     if s(p) and e[k-1] are negligible and k<p
        // kase = 2     if s(k) is negligible and k<p
        // kase = 3     if e[k-1] is negligible, k<p, and
        //              s(k), ..., s(p) are not negligible (qr step).
        // kase = 4     if e(p-1) is negligible (convergence).

        for (k = p-2; k >= -1; --k) {
            if (k == -1) {
                break;
            }
            if (std::fabs(e[k]) <= eps*(std::fabs(singular[k]) +
                std::fabs(singular[k+1]))) {
                    e[k] = 0.0;
                    break;
            }
        }
        if (k == p-2) {
            kase = 4;
        } else {
            Integer ks;
            for (ks = p-1; ks >= k; --ks) {
                if (ks == k) {
                    break;
                }
                Real_d t = (ks != p ? std::fabs(e[ks]) : 0.) +
                    (ks != k+1 ? std::fabs(e[ks-1]) : 0.);
                if (std::fabs(singular[ks]) <= eps*t)  {
                    singular[ks] = 0.0;
                    break;
                }
            }
            if (ks == k) {
                kase = 3;
            } else if (ks == p-1) {
                kase = 1;
            } else {
                kase = 2;
                k = ks;
            }
        }
        k++;

        // Perform the task indicated by kase.

        switch (kase) {

            // Deflate negligible s(p).

              case 1: {
                  Real_d f = e[p-2];
                  e[p-2] = 0.0;
                  for (j = p-2; j >= k; --j) {
                      Real_d t = hypoten(singular[j],f);
                      Real_d cs = singular[j]/t;
                      Real_d sn = f/t;
                      singular[j] = t;
                      if (j != k) {
                          f = -sn*e[j-1];
                          e[j-1] = cs*e[j-1];
                      }
                      for (i = 0; i < n_; i++) {
                          t = cs*V_mat[i][j] + sn*V_mat[i][p-1];
                          V_mat[i][p-1] = -sn*V_mat[i][j] + cs*V_mat[i][p-1];
                          V_mat[i][j] = t;
                      }
                  }
                      }
                      break;

                      // Split at negligible s(k).

              case 2: {
                  Real_d f = e[k-1];
                  e[k-1] = 0.0;
                  for (j = k; j < p; j++) {
                      Real_d t = hypoten(singular[j],f);
                      Real_d cs = singular[j]/t;
                      Real_d sn = f/t;
                      singular[j] = t;
                      f = -sn*e[j];
                      e[j] = cs*e[j];
                      for (i = 0; i < m_; i++) {
                          t = cs*U_mat[i][j] + sn*U_mat[i][k-1];
                          U_mat[i][k-1] = -sn*U_mat[i][j] + cs*U_mat[i][k-1];
                          U_mat[i][j] = t;
                      }
                  }
                      }
                      break;

                      // Perform one qr step.

              case 3: {

                  // Calculate the shift.
                  Real_d scale = std::max(
                      std::max(
                      std::max(
                      std::max(std::fabs(singular[p-1]),
                      std::fabs(singular[p-2])),
                      std::fabs(e[p-2])),
                      std::fabs(singular[k])),
                      std::fabs(e[k]));
                  Real_d sp = singular[p-1]/scale;
                  Real_d spm1 = singular[p-2]/scale;
                  Real_d epm1 = e[p-2]/scale;
                  Real_d sk = singular[k]/scale;
                  Real_d ek = e[k]/scale;
                  Real_d b = ((spm1 + sp)*(spm1 - sp) + epm1*epm1)/2.0;
                  Real_d c = (sp*epm1)*(sp*epm1);
                  Real_d shift = 0.0;
                  if ((b != 0.0) | (c != 0.0)) {
                      shift = std::sqrt(b*b + c);
                      if (b < 0.0) {
                          shift = -shift;
                      }
                      shift = c/(b + shift);
                  }
                  Real_d f = (sk + sp)*(sk - sp) + shift;
                  Real_d g = sk*ek;

                  // Chase zeros.

                  for (j = k; j < p-1; j++) {
                      Real_d t = hypoten(f,g);
                      Real_d cs = f/t;
                      Real_d sn = g/t;
                      if (j != k) {
                          e[j-1] = t;
                      }
                      f = cs*singular[j] + sn*e[j];
                      e[j] = cs*e[j] - sn*singular[j];
                      g = sn*singular[j+1];
                      singular[j+1] = cs*singular[j+1];
                      for (i = 0; i < n_; i++) {
                          t = cs*V_mat[i][j] + sn*V_mat[i][j+1];
                          V_mat[i][j+1] = -sn*V_mat[i][j] + cs*V_mat[i][j+1];
                          V_mat[i][j] = t;
                      }
                      t = hypoten(f,g);
                      cs = f/t;
                      sn = g/t;
                      singular[j] = t;
                      f = cs*e[j] + sn*singular[j+1];
                      singular[j+1] = -sn*e[j] + cs*singular[j+1];
                      g = sn*e[j+1];
                      e[j+1] = cs*e[j+1];
                      if (j < m_-1) {
                          for (i = 0; i < m_; i++) {
                              t = cs*U_mat[i][j] + sn*U_mat[i][j+1];
                              U_mat[i][j+1] = -sn*U_mat[i][j] + cs*U_mat[i][j+1];
                              U_mat[i][j] = t;
                          }
                      }
                  }
                  e[p-2] = f;
                  iter = iter + 1;
                      }
                      break;

                      // Convergence.

              case 4: {

                  // Make the singular values positive.

                  if (singular[k] <= 0.0) {
                      singular[k] = (singular[k] < 0.0 ? -singular[k] : 0.0);
                      for (i = 0; i <= pp; i++) {
                          V_mat[i][k] = -V_mat[i][k];
                      }
                  }

                  // Order the singular values.

                  while (k < pp) {
                      if (singular[k] >= singular[k+1]) {
                          break;
                      }
                      std::swap(singular[k], singular[k+1]);
                      if (k < n_-1) {
                          for (i = 0; i < n_; i++) {
                              std::swap(V_mat[i][k], V_mat[i][k+1]);
                          }
                      }
                      if (k < m_-1) {
                          for (i = 0; i < m_; i++) {
                              std::swap(U_mat[i][k], U_mat[i][k+1]);
                          }
                      }
                      k++;
                  }
                  iter = 0;
                  --p;
                      }
                      break;
        }
    }

    // ok we've got the SVD decomp, we still need to solve

    if (target!=0)
    {


        // set e = U^{t} target
        for (int i=0; i < U_mat.columns(); ++i)
        {
            e[i]=0;

            for (int j=0; j < U_mat.rows(); ++j)
                e[i] += U_mat[j][i]*target[j];

        }

        // divide by singular values 

        for (int i=0; i < n_; ++i)
        {
            if (fabs(singular[i]/singular[0]) >= tolerance) 
                e[i]/= singular[i];
            else 
                e[i] =0.0;
        }

        // multiply by V

        for (int i=0; i <   V_mat.rows(); ++i)
        {
            solution[i]=0.0;

            for (int j=0; j < V_mat.columns(); ++j)
                solution[i]+= V_mat(i,j)*e[j];
        }

    }
}



