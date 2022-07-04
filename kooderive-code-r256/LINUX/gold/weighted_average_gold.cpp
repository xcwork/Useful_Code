
// (c) Mark Joshi 2012
// This code is released under the GNU public licence version 3

//                          weighted_average_gold.cpp

/*
weighted basket example

*/

#include <gold/weighted_average_gold.h>

namespace
{
    void test()
    {
        int stocks =10;
        int paths =1000;
        int steps = 7;
        std::vector<float> inputs_vec(paths);
        std::vector<float> outputs_vec(paths);
        std::vector<float> weights_vec(stocks);

        CubeConstFacade<float> input_mat(&inputs_vec[0],paths,stocks,steps);
        MatrixFacade<float> output_mat(&outputs_vec[0],paths,steps);

        basketWeightings_gold<float>(input_mat, 
            output_mat, 
            weights_vec,
            paths,
            stocks,
            steps);
    }

}
