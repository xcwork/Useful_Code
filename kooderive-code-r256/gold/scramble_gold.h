//
//
//                                              scramble_gold.h
//
//
// (c) Mark Joshi 2010
// This code is released under the GNU public licence version 3

#ifndef SCRAMBLE_GOLD_H
#define SCRAMBLE_GOLD_H

#include <vector>
// dimensions \times paths
void scramble_gold(const std::vector<unsigned int>& input_data_vec, // data to scramble
                   std::vector<unsigned int>& out_data_vec, //  scrambled data
                   const std::vector<unsigned int>& scramblers_vec,
                   int dimensions,
                   int number_paths);

//  paths \times dimensions
void scramble_path_dimension_gold(const std::vector<unsigned int>& input_data_vec, // data to scramble
                                  std::vector<unsigned int>& out_data_vec, //  scrambled data
                                  const std::vector<unsigned int>& scramblers_vec,
                                  int dimensions,
                                  int number_paths);

#endif
