//
//
//                                                                             MultiD_path_gen_BS_gold.h
//
//

// (c) Mark Joshi 2010, 2012,2014
// This code is released under the GNU public licence version 3

#ifndef MULTID_PATH_GEN_BS_GOLD_H
#define MULTID_PATH_GEN_BS_GOLD_H
#include <vector>
#include <gold/MatrixFacade.h> 
#include <gold/Errors.h> 
#include <algorithm>
#include <cmath>

// out puts log rates in outputPaths_vec which is resized appropriately
// uses cubefacade outputPaths_cube(s,r,p)
// s = step
// r = asset index
// p = path number
/*
template<class D>
void multi_dim_BS_path_generator_log_gold(const std::vector<D>& logRates_vec,
                                                                                           const std::vector<D>& correlatedBrownianIncrements_vec,
                                                                                           const std::vector<D>& fixedDrifts_vec,
                                                                                           std::vector<float>& outputPaths_vec, 
                                                                                           int paths,
                                                                                           int rates,
                                                                                           int steps);
                                                                                           */

// out puts  rates in outputPaths_vec which is resized appropriately
template<class D>
void multi_dim_BS_path_generator_nonlog_gold(const std::vector<D>& logRates_vec,
                                                                                           const std::vector<D>& correlatedBrownianIncrements_vec,
                                                                                           const std::vector<D>& fixedDrifts_vec,
                                                                                           std::vector<D>& outputPaths_vec, 
                                                                                           int paths,
                                                                                           int rates,
                                                                                           int steps);

template<class D>
void multi_dim_BS_path_generator_log_gold(const std::vector<D>& logRates_vec,
                                                                                           const std::vector<D>& correlatedBrownianIncrements_vec,
                                                                                           const std::vector<D>& fixedDrifts_vec,
                                                                                           std::vector<D>& outputPaths_vec, 
                                                                                           int paths,
                                                                                           int rates,
                                                                                           int steps)
{
    if (logRates_vec.size()    != rates)
            GenerateError("Log rates array missized on entry to multi_dim_BS_path_generator_gold.");

    if (fixedDrifts_vec.size() !=rates*steps)
        GenerateError("fixedDrifts_vec missized.");

    if (correlatedBrownianIncrements_vec.size() != paths*rates*steps)
        GenerateError("correlatedBrownianIncrements_vec missized.");

    outputPaths_vec.resize(paths*rates*steps); 

    CubeFacade<D> outputPaths_cube(&outputPaths_vec[0],steps,rates,paths);
    CubeConstFacade<D> correlatedBrownianIncrements_cube(&correlatedBrownianIncrements_vec[0],steps,rates,paths);
    MatrixConstFacade<D> drifts_matrix(&fixedDrifts_vec[0],steps,rates);

    for (int p=0; p < paths; ++p)
    {
        for (int r=0; r < rates; ++r)
             outputPaths_cube(0,r,p) = logRates_vec[r]+correlatedBrownianIncrements_cube(0,r,p)+drifts_matrix(0,r);

        for (int s=1; s < steps; ++s)
            for (int r=0; r < rates; ++r) 
               outputPaths_cube(s,r,p) = outputPaths_cube(s-1,r,p) + correlatedBrownianIncrements_cube(s,r,p)+drifts_matrix(s,r);
            
        
    }
    
}

template<class D>
void multi_dim_BS_path_generator_nonlog_gold(const std::vector<D>& logRates_vec,
                                                                                           const std::vector<D>& correlatedBrownianIncrements_vec,
                                                                                           const std::vector<D>& fixedDrifts_vec,
                                                                                           std::vector<D>& outputPaths_vec, 
                                                                                           int paths,
                                                                                           int rates,
                                                                                           int steps)
{
     multi_dim_BS_path_generator_log_gold<D>( logRates_vec,
                                                                                           correlatedBrownianIncrements_vec,
                                                                                           fixedDrifts_vec,
                                                                                           outputPaths_vec, 
                                                                                           paths,
                                                                                           rates,
                                                                                           steps);

     for (int i=0; i < static_cast<int>(outputPaths_vec.size()); ++i)
         outputPaths_vec[i] = exp( outputPaths_vec[i] );

}

#endif
