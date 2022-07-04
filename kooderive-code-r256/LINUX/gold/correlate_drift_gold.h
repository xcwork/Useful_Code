//
//
//                                                                             Correlate_drift_gold.h
//
//

// (c) Mark Joshi 2010,2014
// This code is released under the GNU public licence version 3


#ifndef CORRELATE_DRIFT_GOLD_H
#define CORRELATE_DRIFT_GOLD_H

#include <vector>
#include <gold/MatrixFacade.h>
/*
    Correlate random numbers by multiplying with the Matrix A and add drift
    // input is steps, paths, factors 
*/

template<class D>
void correlate_drift_paths_gold(const std::vector<D>& input_data_host, // randon numbers
                                std::vector<D>& output_data_host, // correlated rate increments 
                                const std::vector<D>& A_vec, // correlator 
                                int AOffsetPerStep, 
                                const std::vector<D>& drift_vec, // drifts 
                                int factors, 
                                int out_dimensions,
                                int number_paths,
                                int steps)
                                



{

    MatrixConstFacade<D> drifts_matrix(&drift_vec[0],steps,  out_dimensions);
 
    for (int i=0; i < number_paths; ++i)
    {
        for (int stepNumber =0; stepNumber< steps;++stepNumber)
        {       
            int offset = i*factors + stepNumber*factors*number_paths;

            int outoffset = i + stepNumber * number_paths*out_dimensions;

            for (int r=0; r < out_dimensions; ++r)
            {
                D total =0.0;
                for (int f=0; f < factors; ++f)
                    total += A_vec[stepNumber*AOffsetPerStep+r*factors+f] * input_data_host[offset+f];

                total+= drifts_matrix(stepNumber,r);

                output_data_host[r*number_paths+outoffset]= total;
            }

        }
    }

}

#endif
