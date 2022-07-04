// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3
// note derived from code provided by NVIDIA 

#ifndef OUTER_PRODUCT_GOLD_H
#define OUTER_PRODUCT_GOLD_H


#include <gold/MatrixFacade.h>

template<class T>
void ReducedOuterProductSymmetric(int paths,
                         int row_size,
                         const MatrixConstFacade<T>& input_data_mat, 
                         MatrixFacade<T>& result_mat)
{
    for (int i=0; i< row_size; ++i)
        for (int j=0; j <= i; ++j)
        {
            double res=0.0;

            for (int p=0; p < paths; ++p)
                res += input_data_mat(i,p)*input_data_mat(j,p);

            result_mat(i,j)=static_cast<T>(res);
            result_mat(j,i)=static_cast<T>(res);
        }
}


template<class T>
void ReducedOuterProduct(int paths,
                         int rows,
                         int cols,
                         const MatrixConstFacade<T>& input_data_mat1, 
                         const MatrixConstFacade<T>& input_data_mat2, 
                         MatrixFacade<T>& result_mat)
{
    for (int i=0; i< rows; ++i)
        for (int j=0; j < cols; ++j)
        {
            double res=0.0;

            for (int p=0; p < paths; ++p)
                res += input_data_mat1(i,p)*input_data_mat2(j,p);

            result_mat(i,j)=static_cast<T>(res);
        }
}


#endif
