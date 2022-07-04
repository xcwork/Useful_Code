//
//
//                                  scramble_gold.cpp
//
// (c) Mark Joshi 2010,2011
// This code is released under the GNU public licence version 3

#include <gold/scramble_gold.h>
#include <gold/MatrixFacade.h> 
#include <vector>
void scramble_gold(const std::vector<unsigned int>& input_data_vec, // data to scramble
                   std::vector<unsigned int>& out_data_vec, //  scrambled data
                   const std::vector<unsigned int>& scramblers_vec,
                   int dimensions,
                   int number_paths)
{
    MatrixConstFacade<unsigned int> data_in(&input_data_vec[0],dimensions, number_paths);
    MatrixFacade<unsigned int> data_out(&out_data_vec[0],dimensions, number_paths);


    for (int p=0; p < number_paths; ++p)
        for (int d=0; d < dimensions; ++d)
            data_out(d,p)  = data_in(d,p)^scramblers_vec[d];



}

void scramble_path_dimension_gold(const std::vector<unsigned int>& input_data_vec, // data to scramble
                                  std::vector<unsigned int>& out_data_vec, //  scrambled data
                                  const std::vector<unsigned int>& scramblers_vec,
                                  int dimensions,
                                  int number_paths)
{
    MatrixConstFacade<unsigned int> data_in(&input_data_vec[0],dimensions, number_paths);
    MatrixFacade<unsigned int> data_out(&out_data_vec[0],dimensions, number_paths);


    for (int p=0; p < number_paths; ++p)
        for (int d=0; d < dimensions; ++d)
            data_out(p,d)  = data_in(p,d)^scramblers_vec[d];

}
