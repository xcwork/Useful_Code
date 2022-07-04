//
//
//                      transpose_gold.cpp
//
//

#include <gold/math/transpose_gold.h>
#include <gold/MatrixFacade.h>

void TransposeMatrix(const MatrixConstFacade<float>& input, std::vector<float>& output_vec)
{
    output_vec.resize(input.columns()*input.rows());
    MatrixFacade<float> output_mat(output_vec,input.columns(),input.rows());

    for (int r=0; r < input.rows();++r)
        for (int c=0; c<input.columns(); ++c)
            output_mat(c,r) = input(r,c);

}