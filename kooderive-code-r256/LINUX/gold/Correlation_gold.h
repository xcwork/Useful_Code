//
//
//                                                                             Correlation_gold.h
//
//

// (c) Mark Joshi 2009,2014
// This code is released under the GNU public licence version 3


#ifndef CORRELATION_GOLD_H
#define CORRELATION_GOLD_H

#include <vector>

/*
    Correlate random numbers by multiplying with the Matrix A. 
*/
template<class D>
void correlated_paths_gold(const std::vector<D>& input_data_host, // randon numbers
                           std::vector<D>& output_data_host, // correlated rate increments 
                           const std::vector<D>& A_vec, // correlator 
                                                                                                 int factors, 
                                                                                                 int out_dimensions,
                                                                                                 int number_paths,
                                                                                                 int stepNumber)

{

 
   for (int i=1; i < number_paths; ++i)
   {
       int offset = i*factors + stepNumber*factors*number_paths;
           // i + stepNumber * number_paths*factors;
       int outoffset = i + stepNumber * number_paths*out_dimensions;
       for (int r=0; r < out_dimensions; ++r)
       {
           D total =0.0;
           for (int f=0; f < factors; ++f)
                total += A_vec[r*factors+f] * input_data_host[offset+f];

           output_data_host[r*number_paths+outoffset]= total;
       }
   }

}

#endif
