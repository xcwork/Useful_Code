#include <gold/math/outerProduct_gold.h>

// to force compilation
namespace
{
    void Test(){

        int paths =1000;
        int rows =10;

        std::vector<float> tmp(paths);
        std::vector<float> tmp2(paths);

        MatrixConstFacade<float> input_data_mat(&tmp[0],paths,rows);
        MatrixFacade<float> result_mat(&tmp[0],rows,rows);


        ReducedOuterProductSymmetric(paths,
            rows,
            input_data_mat, 
            result_mat);


          ReducedOuterProduct(paths,
            rows,
            rows,
            input_data_mat, 
            input_data_mat,   
            result_mat);
    }
}
